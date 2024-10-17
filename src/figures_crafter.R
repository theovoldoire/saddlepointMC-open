library(readr)
library(tidyverse)
library(data.table)
library(latex2exp)
library(ggpubr)

#### SYNTHETIC DATA ####

#### Figure 1 ####

dt <- read_csv("data/for_figures/comparison_standard_error.csv") |> 
  as.data.table()
colnames(dt) = as.character(1:5)

dt2 = melt(dt, id.vars = "1")
dt2$N = as.character(dt2$`1`)
dt2$mode = "With tilting"
dt2[variable == "2" | variable == "4", ]$mode= "Without tilting"
dt2$sampling = "Normal"
dt2[variable == "2" | variable == "3"]$sampling = "Uniform"

dt2$sampling = factor(dt2$sampling, c("Uniform", "Normal"))
dt2$mode = factor(dt2$mode, c("Without tilting", "With tilting"))

ggplot(dt2, aes(x = N, y = value, group=1)) + 
  geom_point() + geom_line() + 
  facet_grid(rows=vars(mode), cols = vars(sampling)) + 
  scale_y_log10("Standard error of the likelihood (log-scale)", 
                breaks=10^c(-10:2)) + 
  scale_x_discrete(TeX("Parameter $n$"))
ggsave("docs/figures/comparison_standard_error.pdf",  height = 8, width = 12, units = "cm")

# Ajouter le rd

#### Figure 3 ####

sparsity <- read_csv("data/for_figures/sparsity.csv")
sparsity$Asymetry = factor(as.character(sparsity$Asymetry), c("1", "2", "5", "10"))
sparsity$Type = factor(ifelse(sparsity$Type == "X", "X close/far from Gaussian", "AX close/far form Gaussian"), levels=c("X close/far from Gaussian", "AX close/far form Gaussian"))

ggplot(sparsity, aes(x = Asymetry, y = Sd, group=1)) + 
  scale_y_log10("Standard deviation of log-likelihood") + 
  geom_point() + geom_line() + facet_grid(cols= vars(Type))

ggsave("docs/figures/sparsity.pdf", height = 8, width=12, units="cm")


#### Figure 4 ####

dt = read_csv("data/for_figures/tail_behavior.csv")

ggplot(dt, aes(x = Observation, y=Value, group=1)) + facet_grid(cols=vars(Tilting), scales="free_y") + 
  geom_point() + geom_line() + scale_y_log10("Standard error of log-likelihood (log-scale)")
ggsave("docs/figures/tail_behavior.pdf", height=8, width=12, units="cm")


#### REAL DATA ####

#### Figure 5 Gaussian comparison ####

dt = read_csv("data/for_figures/comparison_gaussian.csv")

dt2 = dcast(as.data.table(dt), vote_t1 + model + vote_t2 ~ confidence, value.var = "value") |> as.data.frame()
dt2 = dt2[!dt2$vote_t1 %in% c("Royal (left)", "Sarkozy (right)"),]

ggplot(dt2[dt2$model == "True model",], aes(x = `0.5`, xmin = `0.05`, xmax = `0.95`, y = vote_t2)) + 
  facet_grid(rows= vars(vote_t1), scales = "free", switch="both") + 
  geom_errorbar(width=.7) +  
  geom_errorbar(data = dt2[dt2$model != "True model",], aes(x = `0.5`, xmin = `0.5`, xmax = `0.5`, y = vote_t2), col ="red", width=.7) + 
  scale_y_discrete("", position="right") + scale_x_continuous("Posterior") +  theme(
  strip.text = element_text(size = 5),
  axis.text.y = element_text(size= 6)
)

ggsave("docs/figures/comparison_gaussian.pdf",  height = 8, width = 12, units = "cm")

#### Figure 5B Baseline comparison ####

#### Figure 6 Department comparison ####

dt = read_csv("data/for_figures/idf_departments.csv")

dt2 = dcast(as.data.table(dt), vote_t2 + vote_t1 + department ~ quantile, value.var = "value") |>
  as.data.frame()
dt2$department = as.character(dt2$department)
dt2 = dt2[dt2$vote_t2 == "Macron (center)",]
ggplot(dt2, aes(x = department, y = `0.5`, ymin = `0.05`, ymax = `0.95`)) + facet_grid(cols = vars(vote_t2)) +
  geom_errorbar()

df = dt2[,c("department", "0.5")]
colnames(df) = c("dept_code", "value")

# Load shapefile for French departments (using a common shapefile repository for example)
# Assuming you have the shapefile for France departments
url <- "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
france_depts <- st_read(url)

# Filter to include only departments in Île-de-France
ile_de_france_codes <- c("75", "77", "78", "91", "92", "93", "94", "95")
ile_de_france_map <- france_depts %>%
  filter(code %in% ile_de_france_codes)

# Merge map data with department data
ile_de_france_map <- ile_de_france_map %>%
  left_join(df, by = c("code" = "dept_code"))

# Plot the map using ggplot2
ggplot(data = ile_de_france_map) +
  geom_sf(aes(fill = value), color = "black") +
  labs(title = "Department Data for Île-de-France", fill = "Proportion") +
  theme_minimal() + ggtitle("") + theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )

ggsave("docs/figures/comparison_depts.pdf", height=8, width=12, units="cm")

#### Figure 7 - Effect of density ####


dt = read_csv("data/for_figures/density.csv")

ggplot(dt, aes(x = grille, y = `.5`, ymin = `.05`, ymax = `.95`, fill = candidate, group=candidate)) + 
  geom_line() + geom_ribbon() + scale_x_log10("Population density") + scale_y_continuous("Probability") + scale_fill_discrete("Option") + theme(legend.text = element_text(size=6))


ggplot(dt, aes(x = grille, y = `.5`, ymin = `.05`, ymax = `.95`)) +  geom_ribbon(fill = "grey")+
  geom_line(col = "black")  + scale_x_log10("Population density") + scale_y_continuous("Probability")  + 
  facet_grid(rows = vars(candidate), scales = "free") + theme(strip.text = element_text(size = 6))


ggsave("docs/figures/density.pdf", height=8, width=12, units="cm")

 
#### Figure 8 - Borne-Ruffin ####

dt = read_csv("data/for_figures/borne_ruffin.csv")

dt2 = dcast(as.data.table(dt), vote_t1  + vote_t2 + circo ~ confidence, value.var = "value") |> as.data.frame()

ggplot(dt2[dt2$circo==1,], aes(x = `0.5`, xmin = `0.05`, xmax = `0.95`, y = vote_t2)) + 
  facet_grid(rows= vars(vote_t1), scales = "free", switch="both") + 
  geom_errorbar(width=.7) + 
  scale_y_discrete("", position="right") + scale_x_continuous("Posterior") +  theme(
    strip.text = element_text(size = 5),
    axis.text.y = element_text(size= 6)
  )

tab1 = ggplot(dt2[dt2$circo==1,], aes(y = `0.5`, ymin = `0.05`, ymax = `0.95`, x = vote_t2)) + 
  facet_grid(rows= vars(vote_t1), switch="y") + 
  scale_x_discrete("") +
  geom_errorbar(width=.2) + scale_y_continuous("Posterior") +  theme(
    strip.text = element_text(size = 5),
    axis.text.x = element_text(size= 7, angle = 10, vjust = .5),
    axis.text.y = element_text(size=5)
)

tab2 = ggplot(dt2[dt2$circo==2,], aes(y = `0.5`, ymin = `0.05`, ymax = `0.95`, x = vote_t2)) + 
  facet_grid(rows= vars(vote_t1), switch="y") + 
  scale_x_discrete("") +
  geom_errorbar(width=.2) + scale_y_continuous("Posterior") +  theme(
    strip.text = element_text(size = 5),
    axis.text.x = element_text(size= 7, angle = 10, vjust = .5),
    axis.text.y = element_text(size=5)
  )


ggarrange(tab1, tab2, ncol=1, nrow = 2)

ggsave("docs/figures/borne_ruffin.pdf",  height = 15, width = 12, units = "cm")

#### Figure 9 A ####

dt = read_csv("data/for_figures/outcomes_all_constituencies.csv") |> as.data.table()
dt$wins = if_else(dt$situation%in%c(1,2), "Left first (center-to-left)", "Center first (left-to-center)")
dt$remains = if_else(dt$situation%in%c(1,3), "Weakest leaves", "Weakest remains")
dt2 = dt[,.(m = mean(prob), qmin = mean(prob)-2*sd(prob), qmax = mean(prob)+2*sd(prob), std=sd(prob)),.(wins, remains)]
ggplot(dt) + geom_histogram(aes(x = prob), col="black",fill="white") + facet_grid(rows=vars(wins), cols=vars(remains)) + 
  scale_x_continuous("") + scale_y_continuous("Number of constituencies") #+ 
  
  #geom_vline(data = dt2[c(1,3),], aes(xintercept= m), col = "red") + 
  #geom_vline(data = dt2[c(1,3),], aes(xintercept= qmin), col = "red", linetype = "longdash") +
  #geom_vline(data = dt2[c(1,3),], aes(xintercept= qmax), col = "red", linetype = "longdash") 
ggsave("docs/figures/outcomes_all_constituencies.pdf", height=10, width= 10, units = "cm")

#### Figure 10 A ####

dt = read_csv("data/for_figures/outcomes_all_constituencies.csv") |> as.data.table()
dt$wins = if_else(dt$situation%in%c(1,2), "Center-to-left", "Left-to-center")
dt$wins = factor(dt$wins, levels = c("Left-to-center",  "Center-to-left"))
g1 = ggplot(dt[situation %in% c(1, 3),], aes(x = share_far_right, y = prob)) + geom_point()  + 
  facet_grid(cols=vars(wins)) + 
  geom_smooth(method='lm', formula= y~x, se=F) + 
  scale_x_continuous("First round score, top far-right cand.") + scale_y_continuous("Median predicted probabilities") 
summary(lm(data = dt[situation == 1,], prob ~ share_far_right))
summary(lm(data = dt[situation == 3,], prob ~ share_far_right))
summary(lm(data = dt[situation %in% c(1,2),], prob ~ situation+share_far_right+share_strongest))

#### Figure 10 B ####

db = read_csv("data/for_figures/FP_cases.csv")
db = db[db$situation==1,] |> as.data.table()
db2 = db[,.(m = mean(prob), std = sd(prob), cc = .N),.(type_FP)]
db$type_FP[db$type_FP=="FI"] = "LFI"
db2$type_FP = c("PE", "PCF", "PS", "LFI")
g2 = ggplot(db, aes(x = prob)) + 
  geom_histogram(col = "black", fill= "white", bins=30) +
  facet_grid(rows=vars(type_FP)) +
  
  #geom_vline(data=db2, aes(xintercept = m), col = "blue", linewidth = .7) + 
  #geom_vline(data=db2, aes(xintercept = m - 2*std/sqrt(cc-1)), col = "blue", linetype="longdash", linewidth = .7) + 
  #geom_vline(data=db2, aes(xintercept = m + 2*std/sqrt(cc-1)), col = "blue", linetype = "longdash", linewidth = .7) + 
  
  scale_x_continuous("Median predicted probabilities") + 
  scale_y_continuous("Number of constituencies")


ggarrange(g1, g2, ncol=2)
ggsave("docs/figures/other_hypotheses.pdf", width = 15, height = 10, units="cm")

#### APPENDIX ####



#### Figure A1 ####

evaluate_ESS <- read_csv("data/for_figures/evaluate_ESS.csv")

evaluate_ESS$ESS = evaluate_ESS$value
evaluate_ESS$n = factor(as.character(evaluate_ESS$n), levels= c("50", "1000"))

ggplot(evaluate_ESS, aes(x = n, y = ESS, group=1)) + geom_point() + 
  geom_line() + 
  facet_grid(cols=vars(Type), rows=vars(multiplier))
ggsave("docs/figures/evaluate_ESS.svg", height = 10, width = 15, units = "cm")

