


library(tidyverse)

appendage <- "_time"




# scores <- read.csv("./plots/processed_score.csv")

data <- read.csv(paste("./plots/processed", appendage, ".csv", sep=""))

library(devtools)
library(RColorBrewer)
# install_github("onofriandreapg/aomisc")
library(aomisc)
library(doBy)
library(beeswarm)
library(gridExtra)
library(drc)
library(nlme)
library(splines)
library(mgcv)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(caret)

library(stargazer)
library(dplyr)
library(ggstatsplot)
library(reshape2)
library(ggplot2)
library(scales) 

# scores <- transform(scores,
#   room = as.numeric(room),
#   frame = as.numeric(frame),
#   room.norm = as.numeric(room.norm),
#   frame.norm = as.numeric(frame.norm),
#   mem_size = as.numeric(mem_size)
# )

# scores <- within(scores, rm(X))


# room_summary <- summarySE(scores, measurevar="room", groupvars=c("algo","mem_size"))


# print(summary(scores))
# print(scores_summary)
# scores_T <- melt(scores, id.vars = c("algo", "mem_size", "room", "frame"))




# colors <- brewer.pal(n = 2, name = 'Dark2')
colors <- c("#d51212", "#000be4")
colors <- c("#949494", "#0173b2", "#029e73", "#d55e00", "#cc78bc") # nolint
#

se <- function(x) sqrt(var(x) / length(x))


# process <- preProcess(data, method=c("range"))

# data_norm <- predict(process, data)

# data_norm <- data %>% mutate_at(c("frame", "room", "mem_size"), ~(scale(.) %>% as.vector))



#extract hex color codes for a plot with three elements in ggplot2 

n_algo <- length(unique(data$algo))
hex <- hue_pal()(n_algo)

# when using frames not time uncomment
if (appendage == ""){
  data <- data %>% slice(which(data$frame %% 10000000 == 0))

}

if (appendage == "_time"){
data <- data %>% slice(which(data$frame %% 1000 == 0))

}



#display hex color codes
print("colors")
print(hex)


## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

# data$frame <- data$frame / 1000000000
# data$mem_size <- (data$mem_size - 1)/(8-1)
# data <- transform(data, mem_size = as.character(mem_size))


## set the seed to make your partition reproducible
set.seed(42)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]


data <- transform(data, mem_size = as.integer(mem_size), frame= as.integer(frame))

# model <- lm (room ~ algo + log(frame) + poly(mem_size, 1) + mem_size:algo, data = train)
print("HI")

# model <- lm (room ~ algo + poly(frame, 3) + poly(mem_size, 4) + mem_size:algo, data = train)
model2 <- lm(room ~ algo + log(frame) + poly(frame, 3) + poly(mem_size, 3) + mem_size:algo, data = data)

data <- transform(data, mem_size = as.character(mem_size))

model <- lm(room ~ algo + log(frame) + mem_size, data = data)
model1 <- lm(room ~ algo + log(frame) + mem_size + mem_size:algo, data = data)
# model1 <- glm(room ~ algo + frame + mem_size + mem_size:algo, data = data, family = Gamma())

# data <- transform(data, mem_size = as.numeric(mem_size))



total_pred <- predict(model1, data)
# test_pred <- predict(model, test)
combined <- cbind(data, total_pred)

print(summary(model))

#
ag_true <- aggregate(room ~ algo + mem_size + frame, combined, function(x) c(mean = mean(x), sd = se(x)))
ag_pred <- aggregate(total_pred ~ algo + mem_size + frame, combined, function(x) mean(x))
ag_true <- do.call("data.frame", ag_true)

ag <- merge(ag_true, ag_pred)
rsq <- function(x, y) cor(x, y)^2

print("ROOM LM")

# LR_R <- rsq(test$room, test_pred)
# print(paste("R^2 ", LR_R))


hospital_labeller <- function(variable, value) {
}
capitalize <- function(value) {
  return(paste("Memory Size", value))
}

ag$mem_size = factor(ag$mem_size, levels=c('1', '2', '4', '6', '8', '16', '64', '256'))


if (appendage == ""){
frame_plot <- ggplot(ag, aes(x = frame, y = room.mean, color = factor(algo))) +
  geom_line() +
  geom_line(aes(y = total_pred), linetype = "dashed") +
  # geom_smooth(method = lm, formula = y ~ log(x), se=FALSE,
  # linetype = "dashed") +
  geom_ribbon(aes(ymin = room.mean - room.sd,
  ymax = room.mean + room.sd, fill = factor(algo)),
  colour = NA, alpha = 0.1) +
  facet_wrap(. ~ mem_size, labeller = labeller(mem_size = capitalize)) +
  xlab("Frames (in billion)") +
  ylab("Number of Rooms Visited") +
  guides(fill = "none", colour = guide_legend(nrow = 3)) +
  labs(color = "Sampling Algorithm: ") +
  scale_y_continuous(limits = c(1, 13), breaks = seq(1, 13, 2)) +
  scale_x_continuous(labels = c(0, 0.25, 0.5, 0.75, 1)) +
  theme_light() +
  theme(text = element_text(size = 15), panel.spacing.x = unit(0.5, "lines"), legend.position = "bottom", panel.grid.minor=element_blank())#, panel.grid.major=element_blank())
# theme(legend.key.size = unit(1, 'cm'), #change legend key size
#     legend.key.height = unit(1, 'cm'), #change legend key height
#     , #change legend key width
#     legend.title = element_text(size=14), #change legend title font size
#     legend.text = element_text(size=10)) #change legend text font size
ggsave(file=paste("plots/frameplot", appendage, ".png", sep=""), width=12, height=12, dpi=300)
}

if (appendage == "_time"){

frame_plot <- ggplot(ag, aes(x = frame, y = room.mean, color = factor(algo))) +
  geom_line() +
  geom_line(aes(y = total_pred), linetype = "dashed") +
  # geom_smooth(method = lm, formula = y ~ log(x), se=FALSE,
  # linetype = "dashed") +
  geom_ribbon(aes(ymin = room.mean - room.sd,
  ymax = room.mean + room.sd, fill = factor(algo)),
  colour = NA, alpha = 0.1) +
  facet_wrap(. ~ mem_size, labeller = labeller(mem_size = capitalize)) +
  xlab("Time (hrs)") +
  ylab("Number of Rooms Visited") +
  guides(fill = "none", colour = guide_legend(nrow = 3)) +
  labs(color = "Sampling Algorithm: ") +
  scale_y_continuous(limits = c(1, 13), breaks = seq(1, 13, 2)) +
  scale_x_continuous(labels = c(0, 25, 50, 75, 100), breaks=seq(0,360001, 360000/4)) +
  theme_light() +
  theme(text = element_text(size = 15), panel.spacing.x = unit(0.5, "lines"), legend.position = "bottom", panel.grid.minor=element_blank())#, panel.grid.major=element_blank())
# theme(legend.key.size = unit(1, 'cm'), #change legend key size
#     legend.key.height = unit(1, 'cm'), #change legend key height
#     , #change legend key width
#     legend.title = element_text(size=14), #change legend title font size
#     legend.text = element_text(size=10)) #change legend text font size
ggsave(file=paste("plots/frameplot", appendage, ".png", sep=""), width=12, height=12, dpi=300)
}
print("hhhh")
print(frame_plot)

# print(tab_model(model, model2,
#   dv.labels = c("Model 1", "Model 2"), collapse.se = TRUE,
#   show.ci = FALSE, show.reflvl = FALSE, prefix.labels = "varname",
#   p.style = "stars", file="foo.html"
# ))
thrash <- capture.output(stargazer(model, model1, model2,
  dep.var.labels = "Visited Number of Rooms",
  covariate.labels = NULL, title = "Models", align = TRUE,
  out = paste("./plots/foo", appendage, ".txt", sep=""), type = "latex"
))

thrash <- capture.output(stargazer(model, model1, model2,
  dep.var.labels = "Visited Number of Rooms",
  covariate.labels = NULL, title = "Models", align = TRUE,
  out = paste("./plots/foo", appendage, ".tex", sep=""), type = "latex"
))

matrix_coef <- summary(lm(model1))$coefficients
estimates <- matrix_coef[, 1:2]
# estimates <- estimates[6:8,]
# estimates <- do.call("data.frame", estimates)

# algoPER (alpha:0.6, beta:0.4, B=105)

estimates <- data.frame(
  variable = matrix_coef[, 0],
  estimate = matrix_coef[, 1],
  std = matrix_coef[, 2]
)

res <- resid(model1)

plot(fitted(model1), res)
abline(0,0)

qqnorm(res, cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
qqline(res)

plot(density(res))

write.csv(estimates, paste("./plots/estimates", appendage, ".csv", sep=""))
