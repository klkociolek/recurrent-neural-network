abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output ::VecOrMat{Float32}
    gradient ::Union{Nothing, VecOrMat{Float32}}
    batch_gradient ::Union{Nothing, VecOrMat{Float32}}
    Variable(output;) = new(output, nothing, nothing)
end

mutable struct ScalarOperator{F} <: Operator
    inputs ::Any
    output ::Any
    gradient ::Union{Nothing, VecOrMat{Float32},Float32}
    function ScalarOperator(fun, inputs...)
		return new{typeof(fun)}(inputs, nothing, nothing)
	end
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs ::Any
    output ::Any
    gradient ::Union{Nothing, VecOrMat{Float32},Float32}
    function BroadcastedOperator(fun, inputs...)
       return new{typeof(fun)}(inputs, nothing, nothing) 
    end
end

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end