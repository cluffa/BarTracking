using BSON, Augmentor

BSON.@load "data.bson" images masks

struct TrainData
    images
    masks
end

