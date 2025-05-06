from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, NewType, Protocol, overload


class Routing[T]:
    """This is an auxiliary class that is purposely not a dataclass, and should not be instantiated.
    It should only be used for `__class_getitem__`.
    """

    _force_invariant: Callable[[T], T]


NodeId = NewType('NodeId', str)


class Node[StateT, InputT, OutputT]:
    id: NodeId

    async def run(self, state: StateT, inputs: InputT) -> OutputT:
        raise NotImplementedError


@dataclass
class CallNode[StateT, InputT, OutputT](Node[StateT, InputT, OutputT]):
    id: NodeId
    call: Callable[[StateT, InputT], Awaitable[OutputT]]

    async def run(self, state: StateT, inputs: InputT) -> OutputT:
        return await self.call(state, inputs)


@dataclass
class Interruption[StopT, ResumeT]:
    value: StopT
    next_node: Node[Any, ResumeT, Any]


type AnyRoutingCallable[GraphStateT, GraphOutputT, InputT] = Callable[
    [HandleMaker[GraphStateT, GraphOutputT, InputT]], type[Routing[InputT]]
]


class EmptyNodeFunction[OutputT](Protocol):
    def __call__(self) -> OutputT:
        raise NotImplementedError


class StateNodeFunction[StateT, OutputT](Protocol):
    def __call__(self, state: StateT) -> OutputT:
        raise NotImplementedError


class InputNodeFunction[InputT, OutputT](Protocol):
    def __call__(self, inputs: InputT) -> OutputT:
        raise NotImplementedError


class FullNodeFunction[StateT, InputT, OutputT](Protocol):
    def __call__(self, state: StateT, inputs: InputT) -> OutputT:
        raise NotImplementedError


@overload
def node[OutputT](fn: EmptyNodeFunction[OutputT]) -> Node[Any, None, OutputT]: ...
@overload
def node[InputT, OutputT](
    fn: InputNodeFunction[InputT, OutputT],
) -> Node[Any, InputT, OutputT]: ...
@overload
def node[StateT, OutputT](
    fn: StateNodeFunction[StateT, OutputT],
) -> Node[StateT, None, OutputT]: ...
@overload
def node[StateT, InputT, OutputT](
    fn: FullNodeFunction[StateT, InputT, OutputT],
) -> Node[StateT, InputT, OutputT]: ...


def node(fn: Callable[..., Any]) -> Node[Any, Any, Any]:
    signature = inspect.signature(fn)
    signature_error = "Function may only make use of parameters 'state' and 'inputs'"
    node_id = NodeId(fn.__name__)
    if 'state' in signature.parameters and 'inputs' in signature.parameters:
        assert len(signature.parameters) == 2, signature_error
        return CallNode(id=node_id, call=fn)
    elif 'state' in signature.parameters:
        assert len(signature.parameters) == 1, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn(state))
    elif 'state' in signature.parameters:
        assert len(signature.parameters) == 1, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn(inputs))
    else:
        assert len(signature.parameters) == 0, signature_error
        return CallNode(id=node_id, call=lambda state, inputs: fn())


@dataclass
class GraphBuilder[StateT, InputT, OutputT]:
    # TODO: Should get the following values from __class_getitem__ somehow;
    #   this would make it possible to use typeforms without type errors
    state_type: type[StateT] = field(init=False)
    input_type: type[InputT] = field(init=False)
    output_type: type[OutputT] = field(init=False)

    _start_at: AnyRoutingCallable[StateT, OutputT, InputT] | None = field(
        init=False, default=None
    )
    _simple_edges: list[tuple[Node[StateT, Any, Any], Node[StateT, Any, Any]]] = field(
        init=False, default_factory=list
    )
    _routed_edges: list[
        tuple[Node[StateT, Any, Any], AnyRoutingCallable[StateT, OutputT, Any]]
    ] = field(init=False, default_factory=list)

    def start_at(self, routing: AnyRoutingCallable[StateT, OutputT, InputT]):
        self._start_at = routing

    def edge[T](self, source: Node[StateT, Any, T], destination: Node[StateT, T, Any]):
        self._simple_edges.append((source, destination))

    def edges[NodeInputT, NodeOutputT](
        self,
        node: Node[StateT, NodeInputT, NodeOutputT],
        routing: Callable[
            [HandleMaker[StateT, OutputT, NodeInputT]], type[Routing[NodeOutputT]]
        ],
    ):
        self._routed_edges.append((node, routing))

    def build(self) -> Graph[StateT, InputT, OutputT]:
        # TODO: Build nodes from edges/decisions
        nodes: dict[NodeId, Node[StateT, Any, Any]] = {}
        assert self._start_at is not None, (
            'You must call `GraphBuilder.start_at` before building the graph.'
        )
        return Graph[StateT, InputT, OutputT](
            nodes=nodes,
            start_at=self._start_at,
            edges=[(e[0].id, e[1].id) for e in self._simple_edges],
            routed_edges=[(d[0].id, d[1]) for d in self._routed_edges],
        )

    def _check_output(self, output: OutputT) -> None:
        raise RuntimeError(
            'This method is only included for type-checking purposes and should not be called directly.'
        )


@dataclass
class Graph[StateT, InputT, OutputT]:
    nodes: dict[NodeId, Node[StateT, Any, Any]]

    # TODO: Probably need to tweak the following to actually work at runtime...
    start_at: AnyRoutingCallable[StateT, OutputT, InputT]
    edges: list[tuple[NodeId, NodeId]]
    routed_edges: list[tuple[NodeId, AnyRoutingCallable[StateT, OutputT, Any]]]

    def run(self, state: StateT, inputs: InputT) -> OutputT:
        raise NotImplementedError

    def resume[NodeInputT](
        self,
        state: StateT,
        node: Node[StateT, NodeInputT, Any],
        node_inputs: NodeInputT,
    ) -> OutputT:
        raise NotImplementedError


class HandleMaker[GraphStateT, GraphOutputT, HandleInputT](Protocol):
    def __call__[T](
        self, type_: type[T]
    ) -> Handle[T, GraphStateT, GraphOutputT, HandleInputT, T]:
        return Handle(type_)


@dataclass
class Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, HandleOutputT]:
    _source_type: type[SourceT]
    _transforms: tuple[Callable[[GraphStateT, HandleInputT, Any], Any], ...] = field(
        default=()
    )

    _end: bool = field(init=False, default=False)

    # Note: _route_to must use `Any` instead of `HandleOutputT` in the first argument to keep this type contravariant in
    # HandleOutputT. I _believe_ this is safe because instances of this type should never get mutated after this is set.
    _route_to: Node[GraphStateT, Any, Any] | None = field(init=False, default=None)

    def transform[T](
        self, call: Callable[[GraphStateT, HandleInputT, HandleOutputT], T]
    ) -> Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, T]:
        new_transforms = self._transforms + (call,)
        return Handle(self._source_type, new_transforms)

    def end(
        self: Handle[SourceT, GraphStateT, GraphOutputT, HandleInputT, GraphOutputT],
    ) -> type[SourceT]:
        self._end = True
        return self._source_type

    def route_to(self, node: Node[GraphStateT, HandleOutputT, Any]) -> type[SourceT]:
        self._route_to = node
        return self._source_type
