using FastAI, Metalhead
#using FastAI.DataAugmentation: Image
#using FastAI.Vision: Mask
import CairoMakie; CairoMakie.activate!(type="png")

classes = readlines(open("data/codes.txt"))

#data, blocks = Datasets.loadrecipe(FastAI.Vision.ImageSegmentationFolders(), "data/")

images = Datasets.loadfolderdata(
    "data/images",
    filterfn=isimagefile,
    loadfn=loadfile)

masks = Datasets.loadfolderdata(
    "data/labels",
    filterfn=isimagefile,
    loadfn=f -> loadmask(f, classes))

data = (images, masks)

image, mask = sample = getobs(data, 1);

view(image, 50:55, 50:55)

task = BlockTask(
    (Image{2}(), Mask{2}(classes)),
    (
        ProjectiveTransforms((128, 128), augmentations=augs_projection()),
        ImagePreprocessing(),
        OneHot()
    )
)

#checkblock(task.blocks, sample)

xs, ys = FastAI.makebatch(task, data, 100:102)
showbatch(task, (xs, ys))

describetask(task)

backbone = Models.xresnet18()
model = taskmodel(task, backbone);

lossfn = tasklossfn(task)

traindl, validdl = taskdataloaders(data, task, 16)
optimizer = FastAI.Flux.ADAM()

learner = Learner(model, (traindl, validdl), optimizer, lossfn, ToGPU())

fitonecycle!(learner, 25, 0.0001)

showoutputs(task, learner; n = 4)

savetaskmodel("resnet18-backbone.jld2", task, learner.model, force = true)