using Statistics
include("structure.jl")
include("backward.jl")
include("forward.jl")
include("graph.jl")
include("brodcasted_operators.jl")

struct RNN_
    wxh
    whh
    dense1
end

function update_weights!(graph::Vector, learning_rate::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
            node.batch_gradient ./= batch_size
            node.output .-= learning_rate * node.batch_gradient
            fill!(node.batch_gradient, 0)
        end
    end
end

function train(rnn::RNN_, x::Any, y::Any, epochs, batch_size, learning_rate)
    samples = size(x, 2)
    @time for i in 1:epochs
        epoch_loss = 0.0
        global correct_prediction = 0
        global cumulative = 0

        println("Epoch: ", i)

        for j in 1:samples
            x_train1 = Constant(x[1:196, j])
            x_train2 = Constant(x[197:392, j])
            x_train3 = Constant(x[393:588, j])
            x_train4 = Constant(x[589:end, j])
            y_train = Constant(y[:, j])
            
            h0=Variable(zeros(64), name="h0")
            
            h1 = recurrent(x_train1, rnn.wxh,h0,rnn.whh) |> tanh
            h2 = recurrent(x_train2, rnn.wxh,h1,rnn.whh) |> tanh
            h3 = recurrent(x_train3, rnn.wxh,h2,rnn.whh) |> tanh
            h4 = recurrent(x_train4, rnn.wxh,h3,rnn.whh) |> tanh

            d1 = dense(h4, rnn.dense1) |> identity
            e = cross_entropy_loss(d1, y_train)
            graph= topological_sort(e)

            epoch_loss += forward!(graph)
            backward!(graph)

            if j % batch_size == 0
                update_weights!(graph, learning_rate, batch_size)
            end
        end

        @printf("   Average loss: %.4f\n", epoch_loss / samples)
        @printf("   Train accuracy: %.4f\n", correct_prediction / cumulative)
    end
end

function test(rnn::RNN_, x::Matrix{Float32}, y::Any)
    samples = size(x, 2)
    global correct_prediction
    global cumulative

    for i in 1:samples
            x_test1 = Constant(x[1:196, i])
            x_test2 = Constant(x[197:392, i])
            x_test3 = Constant(x[393:588, i])
            x_test4 = Constant(x[589:end, i])
            y_test = Constant(y[:, i])
            h0=Variable(zeros(64), name="h0")
            h1 = recurrent(x_test1, rnn.wxh,h0,rnn.whh) |> tanh
            h2 = recurrent(x_test2, rnn.wxh,h1,rnn.whh) |> tanh
            h3 = recurrent(x_test3, rnn.wxh,h2,rnn.whh) |> tanh
            h4 = recurrent(x_test4, rnn.wxh,h3,rnn.whh) |> tanh
            d1 = dense(h4, rnn.dense1) |> identity
            e = cross_entropy_loss(d1, y_test)
            graph= topological_sort(e)
        forward!(graph)
    end

    @printf("Test accuracy: %.4f\n\n", correct_prediction / cumulative)
end

function xavier_init(out_dim, in_dim)
    return randn(out_dim, in_dim) * sqrt(2.0 / (out_dim + in_dim))
end
