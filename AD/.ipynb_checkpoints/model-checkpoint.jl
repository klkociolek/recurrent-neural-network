using Statistics
include("structure.jl")
include("backward.jl")
include("forward.jl")
include("graph.jl")
include("operators.jl")

struct RNN_
    wxh::Variable 
    whh::Variable
    b::Variable
    dense1::Variable 
end

function update_weights!(graph::Vector, learning_rate::Float32, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
            node.output .-= learning_rate * (node.batch_gradient ./ batch_size)
            fill!(node.batch_gradient, 0.0)
        end
    end
end

function train(rnn::RNN_, x::Matrix{Float32}, y::Any, settings,rnn_settings)
    
    samples =  size(x, 2)
    if size(x, 1)%rnn_settings.input != 0
        println("Size/input rest of division needs to be 0")
    end
    rnn_cells= Int( size(x, 1)/rnn_settings.input)
    
    x_train = [Constant(zeros(Float32,rnn_settings.input)) for _ in 1:rnn_cells]
    y_train = Constant(zeros(Float32,rnn_settings.output))
    
    @time for i in 1:settings.epochs
        
        epoch_loss::Float32 = 0.0
        global correct_prediction = 0
        global cumulative = 0

        println("Epoch: ", i)

        for j in 1:samples
            
            #for k in 1:rnn_cells
                #@views x_train[k].output .= x[rnn_settings.input * (k - 1) + 1:rnn_settings.input * k, j]
            #end
            @views y_train.output .= y[:, j]
            
            h = Variable(zeros(Float32,rnn_settings.hidden))
            for k in 1:rnn_cells
                @views x_train[k].output .= x[rnn_settings.input * (k - 1) + 1:rnn_settings.input * k, j]
    
                h = recurrent(x_train[k], rnn.wxh, h, rnn.whh, rnn.b) |> tanh
            end

            d1 = dense(h, rnn.dense1) |> identity
            e = cross_entropy_loss(d1, y_train)
            graph= topological_sort(e)

            epoch_loss += forward!(graph)
            backward!(graph)

            if j % settings.batch_size == 0
                update_weights!(graph, Float32(settings.eta), settings.batch_size)
            end
        end

        @printf("   Average loss: %.4f\n", epoch_loss / samples)
        @printf("   Train accuracy: %.4f\n", correct_prediction / cumulative)
    end
end

function test(rnn::RNN_, x::Matrix{Float32}, y::Any,rnn_settings)
    samples = size(x, 2)
    global correct_prediction
    global cumulative
    
    rnn_cells= Int(size(x, 1)/rnn_settings.input)
        
    x_test = [Constant(zeros(rnn_settings.input)) for _ in 1:rnn_cells]
    y_test = Constant(zeros(rnn_settings.output))

    for i in 1:samples
            
            for k in 1:rnn_cells
                @views x_test[k].output .= x[rnn_settings.input * (k - 1) + 1:rnn_settings.input * k, i]
            end
            @views y_test.output .= y[:, i]
            
            h = Variable(zeros(Float32,rnn_settings.hidden))
            for k in 1:rnn_cells
                h = recurrent(x_test[k], rnn.wxh, h, rnn.whh,b) |> tanh
            end
            
            d1 = dense(h, rnn.dense1) |> identity
            e = cross_entropy_loss(d1, y_test)
            graph= topological_sort(e)
            forward!(graph)
    end

    @printf("Test accuracy: %.4f\n\n", correct_prediction / cumulative)
end

function xavier_init(out_dim, in_dim)
    return randn(Float32,out_dim, in_dim) * Float32(sqrt(2.0 / (out_dim + in_dim)))
end




