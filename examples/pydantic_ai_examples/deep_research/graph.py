from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NewType, Protocol


class Routing[T]:
    """This is an auxiliary class that is purposely not a dataclass, and should not be instantiated.
    It should only be used for `__class_getitem__`.
    """

    _force_invariant: Callable[[T], T]


NodeId = NewType('NodeId', str)


class Node[InputT, OutputT]:
    id: NodeId
    _force_input_contravariant: Callable[[InputT], None]
    _force_output_covariant: OutputT


@dataclass
class Interruption[StopT, ResumeT]:
    value: StopT
    next_node: Node[ResumeT, Any]


type AnyRoutingCallable[GraphStateT, GraphOutputT, InputT] = Callable[
    [HandleMaker[GraphStateT, GraphOutputT, InputT]], type[Routing[InputT]]
]


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
    _edges: list[tuple[Node[Any, Any], Node[Any, Any]]] = field(
        init=False, default_factory=list
    )
    _routed_edges: list[
        tuple[Node[Any, Any], AnyRoutingCallable[StateT, OutputT, Any]]
    ] = field(init=False, default_factory=list)

    def start_at(self, routing: AnyRoutingCallable[StateT, OutputT, InputT]):
        self._start_at = routing

    def edge[T](self, source: Node[Any, T], destination: Node[T, Any]):
        self._edges.append((source, destination))

    def routed_edge[NodeInputT, NodeOutputT](
        self,
        node: Node[NodeInputT, NodeOutputT],
        routing: Callable[
            [HandleMaker[StateT, OutputT, NodeInputT]], type[Routing[NodeOutputT]]
        ],
    ):
        self._routed_edges.append((node, routing))

    def build(self) -> Graph[StateT, InputT, OutputT]:
        # TODO: Build nodes from edges/decisions
        nodes: dict[NodeId, Node[Any, Any]] = {}
        assert self._start_at is not None, (
            'You must call `GraphBuilder.start_at` before building the graph.'
        )
        return Graph[StateT, InputT, OutputT](
            nodes=nodes,
            start_at=self._start_at,
            edges=[(e[0].id, e[1].id) for e in self._edges],
            routed_edges=[(d[0].id, d[1]) for d in self._routed_edges],
        )

    def _check_output(self, output: OutputT) -> None:
        raise RuntimeError(
            'This method is only included for type-checking purposes and should not be called directly.'
        )


@dataclass
class Graph[StateT, InputT, OutputT]:
    nodes: dict[NodeId, Node[Any, Any]]

    # TODO: Probably need to tweak the following to actually work at runtime...
    start_at: AnyRoutingCallable[StateT, OutputT, InputT]
    edges: list[tuple[NodeId, NodeId]]
    routed_edges: list[tuple[NodeId, AnyRoutingCallable[StateT, OutputT, Any]]]

    def run(self, state: StateT, inputs: InputT) -> OutputT:
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
    _route_to: Node[Any, Any] | None = field(init=False, default=None)

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

    def route_to(self, node: Node[HandleOutputT, Any]) -> type[SourceT]:
        self._route_to = node
        return self._source_type
