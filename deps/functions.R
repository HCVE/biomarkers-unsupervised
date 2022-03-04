get_power_adjacency <- function(data) {
    adjacency_matrix <- abs(cor(data_all_X,use="p"))^4
    k <- softConnectivity(datE=data,power=5)
    par(mfrow=c(1,2))
    hist(k)
    scaleFreePlot(k, main="Check scale free topology\n")
    return(adjacency_matrix)
}
