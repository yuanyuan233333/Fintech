# Clean the environment 
graphics.off()
rm(list = ls(all = TRUE))

# Please change your working directory 


# Pre-load the packages and model_perf function
source("RPackages.R")
source("model_perf.R")

# Import the dataset 
dataset = read_csv("smaller_dataset.csv")
dataset = dataset[,c(3:22)] # keep financial ratios plus the response variable "status" which takes value 0 if the company has not defaulted and 1 if it has.
dataset = dataset[complete.cases(dataset),]
N=dim(dataset)[1]
dataset=rbind(dataset[dataset$status==0,],dataset[dataset$status==1,])
# We define a metric that provides the relative distance between 
# companies by applying the standardized Euclidean distance between each pair (xi,xj) 
# of institutions feature vectors
dist = as.matrix(dist(scale(dataset[-20])))
g = graph_from_adjacency_matrix(dist, mode = "undirected", weighted = TRUE) # we define the graph 

# We find the MST representation of the graph g
g_mst = mst(g)

label=1:N
label[dataset$status==0]=NA
# Plot the g_mst
V(g_mst)$status = dataset$status
V(g_mst)[status == 1]$color = "firebrick1" # color defaulted companies red
V(g_mst)[status == 0]$color = "white" # color active companies green
plot(g_mst, graph = "nsca",
     vertex.label=label, 
     vertex.size = 3, 
     main = "MST representation of the borrowing companies networks")
sd(dataset[4503,]-dataset[4226,])
sd(dataset[4503,]-dataset[4404,])




