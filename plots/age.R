library(ggplot2)
library(ggridges)
library(tidyverse)

age <- read.csv("./plots/age_proc.csv")

age$count <- as.integer(age$count)
age$frame <- as.numeric(age$frame)

age1 <- head(age, n=500)
age1 <- age1 %>% select(bin, frame, count)

library(ggridges)
library(ggplot2)
library(viridis)
library(hrbrthemes)
# install.packages("hrbrthemes")

# basic example
g2<- ggplot(diamonds, aes(x = price, y = cut, fill=..x..)) +
  geom_density_ridges_gradient(scale = 3, rel_min_height = 0.01) +
    scale_fill_viridis(name = "Temp. [F]", option = "C") +
  theme(legend.position = "none")

  g3<- ggplot(age1, aes(x = bin, y = as.factor(frame), fill=..x.., height=count)) +
  geom_density_ridges_gradient(scale = 3, rel_min_height = 0.01) +
    scale_fill_viridis(name = "Temp. [F]", option = "C") +
  theme(legend.position = "none")
# Plot
g <- ggplot(lincoln_weather, aes(x = "Mean Temperature [F]", y = "Month", fill = ..x..)) +
  geom_density_ridges_gradient(scale = 3, rel_min_height = 0.01) +
  scale_fill_viridis(name = "Temp. [F]", option = "C") +
  labs(title = 'Temperatures in Lincoln NE in 2016') +
  theme_ipsum() +
    theme(
      legend.position="none",
      panel.spacing = unit(0.1, "lines"),
      strip.text.x = element_text(size = 8)
    )
# data <- data.frame(x = 1:5, y = rep(1, 5), height = c(0, 1, 3, 4, 2))
# ggplot(data, aes(x, y, height = height)) + geom_ridgeline()

# iris <- data(iris)
# gp <- ggplot(age1, aes(bin, frame, height=count, group = frame)) + geom_density_ridges(alpha=0.6, stat="binline", bins=20)
# print(gp)


d <- data.frame(
  x = rep(1:5, 3),
  y = c(rep(0, 5), rep(1, 5), rep(2, 5)),
  height = c(0, 1, 3, 4, 0, 1, 2, 3, 5, 4, 0, 5, 4, 4, 1)
)

# gp1 <- ggplot(d, aes(x, y, height = height, group = y)) + 
#   geom_ridgeline(fill = "lightblue")
print(gp)
