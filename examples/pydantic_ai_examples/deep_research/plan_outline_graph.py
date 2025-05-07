"""PlanOutline subgraph.

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

from .graph import Graph, Interruption, Routing, TransformContext
from .nodes import Prompt, TypeUnion
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
    """There is enough information to proceed with handling the user's request."""

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


class ReviseOutline(BaseModel):
    choice: Literal['revise']
    details: str


class ApproveOutline(BaseModel):
    choice: Literal['approve']
    message: str  # message to user describing the research you are going to do


class OutlineStageOutput(BaseModel):
    """Use this if you have enough information to proceed."""

    outline: Outline  # outline of the research
    message: str  # message to show user before beginning research


# Node types
@dataclass
class YieldToHuman:
    message: str


# Graph nodes
handle_user_message = Prompt(
    input_type=MessageHistory,
    output_type=TypeUnion[Refuse | Clarify | Proceed],
    prompt='Decide how to proceed from user message',  # prompt
)

generate_outline = Prompt(
    input_type=GenerateOutlineInputs,
    output_type=Outline,
    prompt='Generate the outline',
)

review_outline = Prompt(
    input_type=ReviewOutlineInputs,
    output_type=TypeUnion[ReviseOutline | ApproveOutline],
    prompt='Review the outline',
)


def transform_proceed(ctx: TransformContext[object, MessageHistory, object]):
    return GenerateOutlineInputs(chat=ctx.inputs, feedback=None)


def transform_clarify(ctx: TransformContext[object, object, Clarify]):
    return Interruption(YieldToHuman(ctx.output.message), handle_user_message)


def transform_outline(ctx: TransformContext[State, object, Outline]):
    return ReviewOutlineInputs(chat=ctx.state.chat, outline=ctx.output)


def transform_revise_outline(
    ctx: TransformContext[State, ReviewOutlineInputs, ReviseOutline],
):
    return GenerateOutlineInputs(
        chat=ctx.state.chat,
        feedback=ExistingOutlineFeedback(
            outline=ctx.inputs.outline, feedback=ctx.output.details
        ),
    )


def transform_approve_outline(
    ctx: TransformContext[object, ReviewOutlineInputs, ApproveOutline],
):
    return OutlineStageOutput(outline=ctx.inputs.outline, message=ctx.output.message)


# Graph
g = Graph.builder(
    state_type=State,
    input_type=MessageHistory,
    output_type=TypeUnion[
        Refuse | OutlineStageOutput | Interruption[YieldToHuman, MessageHistory]
    ],
    start_at=handle_user_message,
)
g.edges(
    handle_user_message,
    lambda h: Routing[
        h(Refuse).end()
        | h(Proceed).transform(transform_proceed).route_to(generate_outline)
        | h(Clarify).transform(transform_clarify).end()
    ],
)
g.edges(
    generate_outline,
    lambda h: Routing[h(Outline).transform(transform_outline).route_to(review_outline)],
)
g.edges(
    review_outline,
    lambda h: Routing[
        h(ReviseOutline).transform(transform_revise_outline).route_to(generate_outline)
        | h(ApproveOutline).transform(transform_approve_outline).end()
    ],
)
# g.edge(
#     source=generate_outline,
#     transform=transform_outline,
#     destination=review_outline,
# )
# g.edges(  # or g.edge?
#     generate_outline,
#     review_outline,
# )
# g.edges(
#     generate_outline,
#     lambda h: Routing[h(Outline).route_to(review_outline)],
# )

graph = g.build()
