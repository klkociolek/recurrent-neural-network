include("graph.jl")

import Base: ^, sin, sum, *, +, -, max, reshape, tanh
import LinearAlgebra: mul!, diagm

recurrent(x::GraphNode, Wxh::GraphNode,h::GraphNode,Whh::GraphNode,b::GraphNode) = BroadcastedOperator(recurrent, x, Wxh,h,Whh,b)
forward(::BroadcastedOperator{typeof(recurrent)}, x, Wxh,h,Whh,b) = Wxh * x + Whh * h .+ b
backward(::BroadcastedOperator{typeof(recurrent)}, x, Wxh, h, Whh,b, g) =  return (Wxh' * g, g * x', Whh' * g, g * h',g)

dense(x::GraphNode, w::GraphNode, b::GraphNode)=BroadcastedOperator(dense, x, w, b)
forward(::BroadcastedOperator{typeof(dense)}, x, w, b) = w * x .+ b
backward(::BroadcastedOperator{typeof(dense)}, x, w, b, g) = return (w' * g, g * x', g)

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) =  x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g,-g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) =  x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) =  x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

cross_entropy_loss( ŷ::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, ŷ, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, ŷ, y) =
    let
        global correct_predictions
        if argmax(ŷ) == argmax(y)
            correct_predictions += 1
        end
        ŷ =  ŷ .- maximum(ŷ)
        ŷ = exp.(ŷ) ./ sum(exp.(ŷ))
        return sum(log.(ŷ) .* y) * - 1.0
    end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, ŷ, y, g) =
    let
        ŷ = ŷ .- maximum(ŷ)
        ŷ = exp.(ŷ) ./ sum(exp.(ŷ))
        return tuple(g .* (ŷ .- y))
    end

tanh(x::GraphNode) = ScalarOperator(tanh, x)
forward(::ScalarOperator{typeof(tanh)}, x) = tanh.(x)
backward(::ScalarOperator{typeof(tanh)}, x, g) = tuple(g .* (1 .- tanh.(x).^2))

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = max.(x, zero(x))
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* (x .> 0))

identity(x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = x
backward(::BroadcastedOperator{typeof(identity)}, x, g) = tuple(g)