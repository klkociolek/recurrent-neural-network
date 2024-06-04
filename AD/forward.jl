include("structure.jl")

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = node.output
compute!(node::Variable) = node.output
function compute!(node::Operator)
    inputs = map(input -> input.output, node.inputs)
    node.output = forward(node, inputs...)
    return node.output
end

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end