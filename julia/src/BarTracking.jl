module BarTracking
    export task, model, RGB, predict
    using FastAI, ColorTypes
    
    task, model = loadtaskmodel("resnet18-backbone.jld2")

end