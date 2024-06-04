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

