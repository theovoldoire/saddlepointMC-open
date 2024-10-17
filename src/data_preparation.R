library(data.table)
library(magrittr)
library(dplyr)
library(tidyverse)
library(readr)
library(readxl)

remove_problems = function(dt1, dt2){
  ww = merge(dt1[,c("code_id", "total")], dt2[,c("code_id", "total")], by = "code_id")
  ww$diff = ww$total.y - ww$total.x
  dt1 = dt1[abs(ww$diff)<=50,]
  dt2 = dt2[abs(ww$diff)<=50,] 
  ww = ww[abs(ww$diff)<=50]
  
  dt2$abstention[ww$diff<0] = dt2$abstention[ww$diff<0] - ww$diff[ww$diff<0]
  dt1$abstention[ww$diff>0] = dt1$abstention[ww$diff>0] + ww$diff[ww$diff>0]
  
  dt2$total[ww$diff<0] = dt2$total[ww$diff<0] - ww$diff[ww$diff<0]
  dt1$total[ww$diff>0] = dt1$total[ww$diff>0] + ww$diff[ww$diff>0]
  return(list(dt1, dt2))
}

#### Présidentielle 2007 ####

load("data/source/presidentielle_2007/donnees2007.RData")

dt1 = res.2007.tour1 |> as.data.table()
dt2 = res.2007.tour2 |> as.data.table()

setnames(dt1, "dept", "code_dept")
setnames(dt2, "dept", "code_dept")
setnames(dt1, "bureau", "code_bureau")
setnames(dt2, "bureau", "code_bureau")
setnames(dt1, "inscrits", "total")
setnames(dt2, "inscrits", "total")

dt1$id_ = paste0(dt1$code_dept, dt1$nom_com, dt1$code_com, dt1$code_bureau)
dt2$id_ = paste0(dt2$code_dept, dt2$nom_com, dt2$code_com, dt2$code_bureau)
dt1 = dt1[dt1$id_ != 25721,]
dt2 = dt2[dt2$id_ != 25721,]
dt1$code_id = order(dt1$id_)
dt2$code_id = order(dt2$id_)

n_candidates = c(12, 2)
name_candidates = list(c("BAYR", "BESA","BOVE", "BUFF", "LAGU",
                         "LEPE", "NIHO", "ROYA", "SARK", "SCHI", "VILL", "VOYN"),
                       c("ROYA", "SARK"))
cc1 = c("code_id","code_dept","total","abstention","blanc", unlist(name_candidates[[1]]))
cc2 = c("code_id","code_dept","total","abstention","blanc", name_candidates[[2]])

dt1 = dt1[order(code_id), ..cc1]
dt2 = dt2[order(code_id), ..cc2]
colnames(dt1) = str_to_lower(colnames(dt1))
colnames(dt2) = str_to_lower(colnames(dt2))

## Handle the change of voters across the two turns 



res= remove_problems(dt1, dt2)
dt1= res[[1]]
dt2 =res[[2]]

## Merge small voting units

seuil=80
dt1a = cbind(dt1[total<=seuil, .(code_id=min(code_id)), by="code_dept"][,c("code_id","code_dept")],
  dt1[total<=seuil, lapply(.SD, sum), by = "code_dept", .SDcols = colnames(dt1)[3:ncol(dt1)]][,-"code_dept"])
dt1_ = rbind(dt1a, dt1[total>seuil,])

dt2a = cbind(dt2[total<=seuil, .(code_id=min(code_id)), by="code_dept"][,c("code_id","code_dept")],
             dt2[total<=seuil, lapply(.SD, sum), by = "code_dept", .SDcols = colnames(dt2)[3:ncol(dt2)]][,-"code_dept"])
dt2_ = rbind(dt2a, dt2[total>seuil])

write_csv(dt1_, "data/source/presidentielle_2007/2007_t1.csv")
write_csv(dt2_, "data/source/presidentielle_2007/2007_t2.csv")

#### Législatives 2024 ####

library(readr)

dt1 <- read_delim("data/source/legislatives_2024/l2024_t1_brut.csv", 
                            delim = ";", escape_double = FALSE, 
                  col_types = cols("Code BV" = col_double(),
                                   "Nuance candidat 15"=col_character(),
                                   "Nuance candidat 16"=col_character(),
                                   "Nuance candidat 17"=col_character(),
                                   "Nuance candidat 18"=col_character(),
                                   "Nuance candidat 19"=col_character(),
                                   "Nom candidat 15"=col_character(),
                                   "Nom candidat 16"=col_character(),
                                   "Nom candidat 17"=col_character(),
                                   "Nom candidat 18"=col_character(),
                                   "Nom candidat 19"=col_character(),
                                   "Prénom candidat 15"=col_character(),
                                   "Prénom candidat 16"=col_character(),
                                   "Prénom candidat 17"=col_character(),
                                   "Prénom candidat 18"=col_character(),
                                   "Prénom candidat 19"=col_character(),
                                   "Sexe candidat 15"=col_character(),
                                   "Sexe candidat 16"=col_character(),
                                   "Sexe candidat 17"=col_character(),
                                   "Sexe candidat 18"=col_character(),
                                   "Sexe candidat 19"=col_character(),
                                   "Voix 15"=col_double(),
                                   "Voix 16"=col_double(),
                                   "Voix 17"=col_double(),
                                   "Voix 18"=col_double(),
                                   "Voix 19"=col_double()), trim_ws = TRUE) #|> as.data.table()
dt2 <- read_delim("data/source/legislatives_2024/l2024_t2_brut.csv", 
                            delim = ";", escape_double = FALSE, trim_ws = TRUE,
                  col_types = cols(`Code BV` = col_double()))# |> as.data.table()


setnames(dt1, "Code département", "code_dept")
setnames(dt2, "Code département", "code_dept")
setnames(dt1, "Code commune", "code_com")
setnames(dt2, "Code commune", "code_com")
setnames(dt1, "Code BV", "code_bureau")
setnames(dt2, "Code BV", "code_bureau")
setnames(dt1, "Inscrits", "total")
setnames(dt2, "Inscrits", "total")
setnames(dt1, "Abstentions", "abstention")
setnames(dt2, "Abstentions", "abstention")
setnames(dt1, "Blancs", "blanc")
setnames(dt2, "Blancs", "blanc")
setnames(dt1, "Nuls", "nul")
setnames(dt2, "Nuls", "nul")


## Getting bureau identifier, merging and removing problematic cases ##
dt2$code_dept = ifelse(substr(dt2$code_dept, 1, 1)=="0",substring(dt2$code_dept, 2),dt2$code_dept)
dt2$code_com = ifelse(substr(dt2$code_com, 1, 1)=="0",substring(dt2$code_com, 2),dt2$code_com)

dt1$code_id = paste0(dt1$code_com, "_", dt1$code_bureau)
dt2$code_id = paste0(dt2$code_com, "_", dt2$code_bureau)

dt1 = dt1[dt1$code_id %in% intersect(dt1$code_id, dt2$code_id),]
dt2 = dt2[dt2$code_id %in% intersect(dt1$code_id, dt2$code_id),]

dt1 = distinct(dt1, code_id, .keep_all = TRUE) |> as.data.table()
dt2 = distinct(dt2, code_id, .keep_all = TRUE) |> as.data.table()

dt1 = dt1[order(dt1$code_id),]
dt2 = dt2[order(dt2$code_id),]

A = remove_problems(dt1, dt2)
dt1 = A[[1]]
dt2 = A[[2]]

## Identifier circonscription ##

columns = c("Numéro de panneau ", "Nuance candidat ", "Nom candidat ", "Prénom candidat ", "Sexe candidat ","Voix ", "% Voix/inscrits " , "% Voix/exprimés ", "Elu ")

cc_name_long = paste0(columns[c(3, 4, 2)], 1)
B = dt1[,..cc_name_long]
dt1$code_circ = paste0(B[[1]], " ", B[[2]], " ", B[[3]]) |> factor() |> as.double()

dt2 = merge(dt1[,c("code_id", "code_circ")], dt2, by = "code_id")


dt1u = list()
for (cand_id in 1:19){
  
  XX = distinct(dt1, code_circ, .keep_all = T)[order(code_circ),]
  cc_name_long = paste0(columns[c(3, 4, 2)], cand_id)
  A1 = XX[,..cc_name_long]
  B1 = paste0(A1[[1]], " ", A1[[2]], " ", A1[[3]])
  
  cc_name = paste0(columns[c(3)], cand_id)
  B2 = XX[[cc_name]]
  
  cc_fname = paste0(columns[c(4)], cand_id)
  B2b = XX[[cc_fname]]
  
  cc_poli = paste0(columns[c(2)], cand_id)
  B3 = XX[[cc_poli]]
  
  cols_voix = paste0(columns[6], cand_id)
  B4 = XX[[cols_voix]]
  
  if (sum(is.na(B3)) < length(B3)){
    dt1u[[cand_id]] = data.frame(code_circ = XX$code_circ, cand_id = cand_id, 
                               name_long=B1, name=B2, fname = B2b, orientation=B3, 
                               code_dept = XX$code_dept)
  }
}

dt2u = list()
for (cand_id in 1:4){
  
  XX = distinct(dt2, code_circ, .keep_all = T)[order(code_circ),]
  cc_name_long = paste0(columns[c(3, 4, 2)], cand_id)
  A1 = XX[,..cc_name_long]
  B1 = paste0(A1[[1]], " ", A1[[2]], " ", A1[[3]])
  
  cc_name = paste0(columns[c(3)], cand_id)
  B2 = XX[[cc_name]]
  
  cc_fname = paste0(columns[c(4)], cand_id)
  B2b = XX[[cc_fname]]
  
  cc_poli = paste0(columns[c(2)], cand_id)
  B3 = XX[[cc_poli]]
  
  cols_voix = paste0(columns[6], cand_id)
  B4 = XX[[cols_voix]]
  
  if (sum(is.na(B3)) < length(B3)){
    dt2u[[cand_id]] = data.frame(code_circ = XX$code_circ, cand_id = cand_id, 
                                 name_long=B1, name=B2, fname = B2b,
                                 orientation=B3, code_dept = XX$code_dept)
  }
}
codex1 = drop_na(rbindlist(dt1u))
codex2 = drop_na(rbindlist(dt2u))

codex1$bloc = case_when(
  codex1$orientation %in% c("MDM", "HOR", "ENS", "DVC") ~ "center", 
  codex1$orientation %in% c("LR", "DVD", "UDI") ~ "right",
  codex1$orientation %in% c("EXG", "COM", "FI", "SOC", "RDG", "VEC", "DVG", "UG", "ECO") ~ "left",
  codex1$orientation %in% c("RN", "DSV", "REC", "UXD", "EXD") ~ "far_right", 
  codex1$orientation %in% c("REG", "DIV") ~ "other"
)

codex2$bloc = case_when(
  codex2$orientation %in% c("MDM", "HOR", "ENS", "DVC") ~ "center", 
  codex2$orientation %in% c("LR", "DVD", "UDI") ~ "right",
  codex2$orientation %in% c("EXG", "COM", "FI", "SOC", "RDG", "VEC", "DVG", "UG", "ECO") ~ "left",
  codex2$orientation %in% c("RN", "DSV", "REC", "UXD", "EXD") ~ "far_right", 
  codex2$orientation %in% c("REG", "DIV") ~ "other"
)

write_csv(codex1, "data/source/legislatives_2024/l2024_codex1.csv")
write_csv(codex2, "data/source/legislatives_2024/l2024_codex2.csv")
# Tidy up the candidates

A1 = dt1[,paste0("Voix ", 1:19)]
colnames(A1) = paste0("V",1:19)
A2 = dt2[,paste0("Voix ", 1:4)]
colnames(A2) = paste0("V",1:4)

dt1$n_cand = rowSums(!is.na(A1))
dt2$n_cand = rowSums(!is.na(A2))
B1 = dt1[,c("code_dept","code_circ", "n_cand","code_id", "total","abstention")]
B2 = dt2[,c("code_dept","code_circ", "n_cand","code_id", "total", "abstention")]
B1$blanc_nul = dt1$blanc + dt1$nul
B2$blanc_nul = dt2$blanc + dt2$nul

dt1b = cbind(B1, A1)[order(code_circ, code_id)]
dt2b = cbind(B2, A2)[order(code_circ, code_id)]

## Merging small units ##

seuil=70
dt1c = cbind(dt1b[total<=seuil, .(code_dept=min(code_dept), code_id=min(code_id), n_cand=min(n_cand)), by="code_circ"][,-c("code_circ")],
             dt1b[total<=seuil, lapply(.SD, sum), by = "code_circ", .SDcols = colnames(dt1b)[5:ncol(dt1b)]])
dt1d = rbind(dt1c, dt1b[total>seuil,])

dt2c = cbind(dt2b[total<=seuil, .(code_dept=min(code_dept), code_id=min(code_id), n_cand=min(n_cand)), by="code_circ"][,-c("code_circ")],
             dt2b[total<=seuil, lapply(.SD, sum), by = "code_circ", .SDcols = colnames(dt2b)[5:ncol(dt2b)]])
dt2d = rbind(dt2c, dt2b[total>seuil,])


write_csv(dt1d, "data/source/legislatives_2024/l2024_T1.csv")
write_csv(dt2d, "data/source/legislatives_2024/l2024_T2.csv")

#### Analysis codex ####

codex1 = read_csv("data/source/legislatives_2024/l2024_codex1.csv") |> as.data.table()
codex2 = read_csv("data/source/legislatives_2024/l2024_codex2.csv") |> as.data.table()

dt1 = read_csv("data/source/legislatives_2024/l2024_T1.csv") |> as.data.table()
dt2 = read_csv("data/source/legislatives_2024/l2024_T2.csv") |> as.data.table()


rr1 = melt(dt1[,lapply(.SD, sum), by = "code_circ", .SDcols = paste0("V",1:19)], id.vars = "code_circ", na.rm = T, variable.name = "cand_id", value.name = "votes")
rr2 = melt(dt2[,lapply(.SD, sum), by = "code_circ", .SDcols = paste0("V",1:4)], id.vars = "code_circ", na.rm = T, variable.name = "cand_id", value.name = "votes")

rr1$cand_id = as.double(substring(rr1$cand_id, 2))
rr2$cand_id = as.double(substring(rr2$cand_id, 2))

rr1 = merge(rr1, codex1[,c("code_circ", "cand_id", "bloc")], by = c("code_circ", "cand_id"))
rr2 = merge(rr2, codex2[,c("code_circ", "cand_id", "bloc")], by = c("code_circ", "cand_id"))

rr1$rank = rr1[,.(rank=rank(-as.double(votes)), cand_id),.(code_circ)]$rank
rr1$share_votes = rr1[,.(share_votes=votes/sum(votes), cand_id),.(code_circ)]$share_votes
rr2$rank = rr2[,.(rank=rank(-as.double(votes)), cand_id),.(code_circ)]$rank

rr1 = rr1[order(code_circ, rank)]
rr2 = rr2[order(code_circ, rank)]

rr1b = dcast(rr1[rank<=3], code_circ ~ rank, value.var = "bloc")
rr2b = dcast(rr2[rank<=3], code_circ ~ rank, value.var = "bloc")

# Now, criteria

crit1a = rr2[,.(any(bloc == "far_right")),.(code_circ)]$V1 
crit1b = rr1[,.(any(rank <= 3 & bloc == "far_right")), .(code_circ)]$V1

crit1 = crit1a & crit1b

colnames(rr1b) = c("code_circ", "r1", "r2", "r3")
colnames(rr2b) = c("code_circ", "r1", "r2", "r3")

crit2a1 = rr1b[,(r1 == "left" & r2 == "center") | (r1 == "left" & r3 == "center") | (r2 == "left" & r3 == "center"),]
crit2b1 = rr1b[,(r1 == "center" & r2 == "left") | (r1 == "center" & r3 == "left") | (r2 == "center" & r3 == "left"),]

crit2a2 = rr2[,.(!any(bloc == "center")),.(code_circ)]$V1
crit2b2 = rr2[,.(!any(bloc == "left")),.(code_circ)]$V1

crit2a = crit2a1 & crit2a2
crit2b = crit2b1 & crit2b2

situation = case_when(
  !crit1 ~ "0. No far right (or not enough)",
  crit1 & (!crit2a1) & (!crit2b1) ~ "0. Not a situation with left and center",
  crit1 & crit2a1 & crit2a2 ~ "1. Left wins, center leaves", 
  crit1 & crit2a1 & !crit2a2 ~ "2. Left wins, center stays",
  crit1 & crit2b1 & crit2b2 ~ "3. Center wins, left leaves",
  crit1 & crit2b1 & !crit2b2 ~ "4. Center wins, left stays",
  T ~ "0. other"
)
situation2 = as.double(substr(situation, 1, 1))

rr1b$situation = situation
rr1b$situation_code = situation2

rr1b = merge(rr1b, rr1[bloc=="far_right",][,.(max(share_votes)),.(code_circ)], by = "code_circ", all.x=T)
setnames(rr1b, "V1", "share_far_right")
rr1b$share_far_right[is.na(rr1b$far_right)] = 0

rr1b = merge(rr1b, rr1[bloc%in%c("left", "center"),][,.(max(share_votes)),.(code_circ)], by = "code_circ", all.x=T)
setnames(rr1b, "V1", "share_strongest")
rr1b$share_far_right[is.na(rr1b$far_right)] = 0


#rr1b$X = rr1$share_votes


write_csv(rr1b[,c("code_circ", "situation_code", "share_far_right", "share_strongest")], "data/source/legislatives_2024/l2024_situation_circos.csv")
# Crit1 : ext droite in the second rounnd
# Crit 2a1 : left won first round
# Crit 2b1 : center won first round
# Crit 2a2 : center not in the second round
# Crit 2b2 : left not in the second round


#### Name of winners legislatives 2024####

remove_accents <- function(text) {
  text <- iconv(text, from = "UTF-8", to = "ASCII//TRANSLIT")
  text <- gsub("[^[:alnum:][:space:][:punct:]]", "", text)
  text <- gsub("`", "", text)
  text <- gsub("'", "", text)
  return(text)
}

codex2 = read_csv("data/source/legislatives_2024/l2024_codex2.csv")
dt2 = read_csv("data/source/legislatives_2024/l2024_T2.csv") |> as.data.table()
rr2 = melt(dt2[,lapply(.SD, sum), by = "code_circ", .SDcols = paste0("V",1:4)], id.vars = "code_circ", na.rm = T, variable.name = "cand_id", value.name = "votes")
rr2$cand_id = as.double(substring(rr2$cand_id, 2))
rr2 = merge(rr2, codex2[,c("code_circ", "cand_id", "bloc", "name_long", "name", "fname", "code_dept")], by = c("code_circ", "cand_id"))
rr2$rank = rr2[,.(rank=rank(-as.double(votes)), cand_id),.(code_circ)]$rank

rr2[,.(votes, rank=rank(-as.double(votes)), cand_id),.(code_circ)]


rr2 = rr2[order(code_circ, rank)][rank==1]


dd = read_csv("data/source/legislatives_2024/liste_deputes.csv")

rr2$nameid = remove_accents(str_to_lower(paste0(
  substr(rr2$name, 1, 4), "_", substr(rr2$fname, 1, 1), as.double(rr2$code_dept))))

dd$nameid = remove_accents(str_to_lower(paste0(
  substr(dd$nom, 1, 4), "_", substr(dd$prenom, 1, 1), as.double(dd$departementCode))))

View(rr2[!nameid %in% dd$nameid,])


res = merge(dd, rr2, by = "nameid")[,c("nameid", "departementNom", "departementCode", "circo", "code_circ", "nom")]

res$name_circ = paste0(res$circo, ifelse(res$circo==1, "ère", "ème"), " ", res$departementNom)
colnames(res) = c("nameid", "name_dept", "code_dept", "number_circ", "code_circ", "name", "name_circ")
res

write_csv(res[,c("code_circ", "name_circ", "name_dept", "code_dept", "number_circ")], "data/source/legislatives_2024/l2024_name_circos.csv")

#### Présidentielle 2022 ####

library(readxl)
dt1_ <- read_excel("data/source/presidentielle_2022/p2022_t1_brut.xlsx") #|> as.data.table()
dt2_ <- read_excel("data/source/presidentielle_2022/p2022_t2_brut.xlsx") #|> as.data.table()
context <- read_excel("data/source/presidentielle_2022/p2022_context_data_brut.xlsx", skip = 5)# |> as.data.table()

dt1 = dt1_
dt2 = dt2_

setnames(dt1, "Code de la circonscription", "code_circ")
setnames(dt2, "Code de la circonscription", "code_circ")
setnames(dt1, "Code du département", "code_dept")
setnames(dt2, "Code du département", "code_dept")
setnames(dt1, "Code de la commune", "code_com")
setnames(dt2, "Code de la commune", "code_com")
setnames(dt1, "Code du b.vote", "code_bureau")
setnames(dt2, "Code du b.vote", "code_bureau")
setnames(dt1, "Inscrits", "total")
setnames(dt2, "Inscrits", "total")
setnames(dt1, "Abstentions", "abstention")
setnames(dt2, "Abstentions", "abstention")
setnames(dt1, "Blancs", "blanc")
setnames(dt2, "Blancs", "blanc")
setnames(dt1, "Nuls", "nul")
setnames(dt2, "Nuls", "nul")


dt1$code_com = paste0(dt1$code_dept, dt1$code_com)
dt2$code_com = paste0(dt2$code_dept, dt2$code_com)
dt1$code_circ = paste0(dt1$code_dept, dt1$code_circ)
dt2$code_circ = paste0(dt2$code_dept, dt2$code_circ)

dt1$code_id = paste0(dt1$code_com, "_", dt1$code_bureau)
dt2$code_id = paste0(dt2$code_com, "_", dt2$code_bureau)

dt1$total = as.double(dt1$total)
dt2$total = as.double(dt2$total)

# Remove problematic cases

dt1 = dt1[!(dt1$code_dept=="ZZ" | dt1$code_circ %in% c("7501","ZW01")),]
dt2 = dt2[!(dt2$code_dept=="ZZ" | dt2$code_circ %in% c("7501","ZW01")),]

dt1 = dt1[order(dt1$code_id),]
dt2 = dt2[order(dt2$code_id),]

ww = merge(dt1[,c("code_id", "total")], dt2[,c("code_id", "total")], by = "code_id")
ww$diff = ww$total.y - ww$total.x
dt1 = dt1[abs(ww$diff)<=50,]
dt2 = dt2[abs(ww$diff)<=50,] 
ww = ww[abs(ww$diff)<=50,]

dt2$abstention[ww$diff<0] = dt2$abstention[ww$diff<0] - ww$diff[ww$diff<0]
dt1$abstention[ww$diff>0] = dt1$abstention[ww$diff>0] + ww$diff[ww$diff>0]

dt2$total[ww$diff<0] = dt2$total[ww$diff<0] - ww$diff[ww$diff<0]
dt1$total[ww$diff>0] = dt1$total[ww$diff>0] + ww$diff[ww$diff>0]

columns = c("N°Panneau ", "Sexe ", "Nom ", "Prénom ", "Voix ", "% Voix/Ins ","% Voix/Exp ")

dt1 = as.data.table(dt1)
dt2 = as.data.table(dt2)

dt1u = list()
for (cand_id in 1:12){
  print(cand_id)
  
  XX = dt1
  cc_name_long = paste0(columns[c(3, 4)], cand_id)
  A1 = XX[,..cc_name_long]
  B1 = paste0(A1[[1]], " ", A1[[2]])
  
  if (sum(is.na(B1)) < length(B1)){dt1u[[cand_id]] = data.frame(cand_id = cand_id, name_long=B1)}
}

dt2u = list()
for (cand_id in 1:2){
  
  XX = dt2
  cc_name_long = paste0(columns[c(3, 4)], cand_id)
  A1 = XX[,..cc_name_long]
  B1 = paste0(A1[[1]], " ", A1[[2]])
  
  if (sum(is.na(B1)) < length(B1)){dt2u[[cand_id]] = data.frame(cand_id = cand_id, name_long=B1)}

}
codex1 = drop_na(rbindlist(dt1u))
codex2 = drop_na(rbindlist(dt2u))

## Tidy up the candidates

A1 = dt1[,paste0("Voix ", 1:12)]
colnames(A1) = paste0("V",1:12)
A2 = dt2[,paste0("Voix ", 1:2)]
colnames(A2) = paste0("V",1:2)

dt1$n_cand = rowSums(!is.na(A1))
dt2$n_cand = rowSums(!is.na(A2))
B1 = dt1[,c("code_dept","code_circ","code_com","n_cand","code_id", "total","abstention")]
B2 = dt2[,c("code_dept","code_circ","code_com","n_cand","code_id", "total", "abstention")]
B1$blanc_nul = dt1$blanc + dt1$nul
B2$blanc_nul = dt2$blanc + dt2$nul

dt1b = cbind(B1, A1)[order(code_id)]
dt2b = cbind(B2, A2)[order(code_id)]

## Merge small bureaux ##

seuil=70
dt1c = cbind(dt1b[total<=seuil, .(code_dept=min(code_dept), code_id=min(code_id), code_com = min(code_com), n_cand=min(n_cand)), by="code_circ"][,-c("code_circ")],
             dt1b[total<=seuil, lapply(.SD, sum), by = "code_circ", .SDcols = colnames(dt1b)[6:ncol(dt1b)]])
dt1d = rbind(dt1c, dt1b[total>seuil,])

dt2c = cbind(dt2b[total<=seuil, .(code_dept=min(code_dept), code_id=min(code_id), code_com = min(code_com), n_cand=min(n_cand)), by="code_circ"][,-c("code_circ")],
             dt2b[total<=seuil, lapply(.SD, sum), by = "code_circ", .SDcols = colnames(dt2b)[6:ncol(dt2b)]])
dt2d = rbind(dt2c, dt2b[total>seuil,])

dt1d = dt1d[code_id %in% dt2d$code_id]
dt2d = dt2d[code_id %in% dt1d$code_id]

### Covariates ###

dt3 = context[,c("CODGEO", "P21_POP", "SUPERF", "MED21")]

setnames(dt3, "CODGEO", "code_com")
dt3$density = log(dt3$P21_POP / dt3$SUPERF)
dt3$density[dt3$density<0] = 0
dt3$median_income = as.double(dt3$MED21)


dt3b = merge(dt2d[,c("code_id","code_com")], dt3, by = "code_com")[order(code_id)]

dt1e = dt1d[code_id %in% dt3b$code_id,][order(code_id)]
dt2e = dt2d[code_id %in% dt3b$code_id,][order(code_id)]

colnames(dt1e)[(ncol(dt1e)-12+1):ncol(dt1e)] = c("Arthaud", "Roussel", "Macron", "Lassalle", "Le Pen", "Zemmour", "Melenchon",
                                                 "Hidalgo", "Jadot", "Pecresse", "Poutou", "Dupont-Aignan")
colnames(dt2e)[(ncol(dt2e)-2+1):ncol(dt2e)] = c("Macron", "Le Pen")

write_csv(dt1e, "data/source/presidentielle_2022/p2022_t1_net.csv")
write_csv(dt2e, "data/source/presidentielle_2022/p2022_t2_net.csv")
write_csv(dt3b, "data/source/presidentielle_2022/p2022_context_net.csv")
