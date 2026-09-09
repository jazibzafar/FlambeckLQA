#install.packages("lidR", repos = c("https://r-lidar.r-universe.dev", "https://cloud.r-project.org"))
#install.packages("RANN")

library(lidR)
library(sf)
library(terra)
library(RANN)

#chm einlesen
chm <- rast("P:\\projects\\Flambeck\\R-Projekte\\Data\\DHM nach gepflanzter Baumarten\\Europ_Lärche.tif")

# Fenstergröße definieren
f <- function(x) {
  
  y <- 2.5 + 0.15 * x
  
  y[x < 2] <- 2.5
  y[x > 25] <- 6
  
  return(y)
}

ttops <- locate_trees(chm, lmf(f))
#gefundene Bäume zählen
nrow(ttops)
#Baumspitzen darstellen
plot(
  chm,
  col = height.colors(50),
  main = "Detektierte Baumspitzen"
)
plot(
  sf::st_geometry(ttops),
  add = TRUE,
  pch = 3,
  col = "red"
)

summary(ttops$Z)

table(cut(
  ttops$Z,
  breaks = c(0, 2, 5, 10, 15, 20, 25, 30, 40, 50),
  include.lowest = TRUE
))

# Abstand zwischen den Baumspitzen
coords <- sf::st_coordinates(ttops)

head(coords)
d <- as.matrix(dist(coords))
nn <- RANN::nn2(coords, k = 2)
summary(nn$nn.dists[,2])

#Wie viel Prozent der Bäume habe welche Größe
quantile(
  ttops$Z,
  probs = c(0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1),
  na.rm = TRUE
)


writeRaster(
  chm,
  "Z:/projects/Flambeck/R-Projekte/Data/Douglasie_export.tif",
  overwrite = TRUE
)