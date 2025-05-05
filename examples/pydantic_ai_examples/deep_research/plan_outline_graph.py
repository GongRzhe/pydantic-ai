"""PlanOutline subgraph

state PlanOutline {
    [*]
    ClarifyRequest: Clarify user request & scope
    HumanFeedback: Human provides clarifications
    GenerateOutline: Draft initial outline
    ReviewOutline: Supervisor reviews outline

    [*] --> ClarifyRequest
    ClarifyRequest --> HumanFeedback: need more info
    HumanFeedback --> ClarifyRequest
    ClarifyRequest --> GenerateOutline: ready
    GenerateOutline --> ReviewOutline
    ReviewOutline --> GenerateOutline: revise
    ReviewOutline --> [*]: approve
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from .graph import GraphBuilder, Interruption, Routing
from .nodes import Prompt
from .shared_types import MessageHistory, Outline


# Types
## State
@dataclass
class State:
    chat: MessageHistory
    outline: Outline | None


## handle_user_message
class Clarify(BaseModel):
    """Ask some questions to clarify the user request."""

    choice: Literal['clarify']
    message: str


class Refuse(BaseModel):
    """Use this if you should not do research.

    This is the right choice if the user didn't ask for research, or if the user did but there was a safety concern.
    """

    choice: Literal['refuse']
    message: str  # message to show user


class Proceed(BaseModel):
    """There is enough information to proceed with handling the user's request"""

    choice: Literal['proceed']


## generate_outline
class ExistingOutlineFeedback(BaseModel):
    outline: Outline
    feedback: str


class GenerateOutlineInputs(BaseModel):
    chat: MessageHistory
    feedback: ExistingOutlineFeedback | None


## review_outline
class ReviewOutlineInputs(BaseModel):
    chat: MessageHistory
    outline: Outline


class OutlineNeedsRevision(BaseModel):
    choice: Literal['revise']
    details: str


class OutlineApproved(BaseModel):
    choice: Literal['approve']
    message: str  # message to user describing the research you are going to do


class OutlineStageOutput(BaseModel):
    """Use this if you have enough information to proceed"""

    outline: Outline  # outline of the research
    message: str  # message to show user before beginning research


# Node types
@dataclass
class YieldToHuman:
    message: str


# Graph nodes
handle_user_message = Prompt[MessageHistory, Refuse | Clarify | Proceed](
    'Decide how to proceed from user message',  # prompt
)

generate_outline = Prompt[GenerateOutlineInputs, Outline](
    'Generate the outline',
)

review_outline = Prompt[ReviewOutlineInputs, OutlineNeedsRevision | OutlineApproved](
    'Review the outline',
)

# Graph
g = GraphBuilder[
    State,
    MessageHistory,
    Refuse | OutlineStageOutput | Interruption[YieldToHuman, MessageHistory],
]()

g.start_at(routing=lambda h: Routing[h(MessageHistory).route_to(handle_user_message)])
g.routed_edge(
    handle_user_message,
    lambda h: Routing[
        h(Refuse).end()
        | h(Proceed)
        .transform(
            call=lambda s, _i, _o: GenerateOutlineInputs(chat=s.chat, feedback=None),
        )
        .route_to(generate_outline)
        | h(Clarify)
        .transform(
            lambda _s, _i, o: Interruption(YieldToHuman(o.message), handle_user_message)
        )
        .end()
    ],
)
g.routed_edge(
    generate_outline,
    lambda h: Routing[
        h(Outline)
        .transform(lambda s, _i, o: ReviewOutlineInputs(chat=s.chat, outline=o))
        .route_to(review_outline)
    ],
)

g.routed_edge(
    review_outline,
    lambda h: Routing[
        h(OutlineNeedsRevision)
        .transform(
            call=lambda s, i, o: GenerateOutlineInputs(
                chat=s.chat,
                feedback=ExistingOutlineFeedback(outline=i.outline, feedback=o.details),
            ),
        )
        .route_to(generate_outline)
        | h(OutlineApproved)
        .transform(
            call=lambda _s, i, o: OutlineStageOutput(
                outline=i.outline, message=o.message
            ),
        )
        .end()
    ],
)
