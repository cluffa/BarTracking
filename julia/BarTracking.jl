using CUDA, Flux, Tracker
using BSON: @load
using Flux: loadmodel!

include("Model.jl")
using .Model

# model = Model.NewModel(3, 2);
# model = loadmodel!(model, @load("model.bson"))

@load "model.bson" model
weights = Tracker.data.(Flux.params(model));
Flux.loadparams!(model, weights)

model = cpu(model)


r = rand(300, 300, 3, 10) |> gpu