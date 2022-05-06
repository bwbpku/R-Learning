#矩阵
x <- matrix(c(15:20),2,3,dimnames = list(c("A","B"),c("x1","x2","x3")))
#数组
y <- array(1:18,c(2,2,3),dimnames = list(c("A","B"),c("x1","x2"),c("y1","y2","y3")))
#数据框
a <- c(1,2,3)
b <- c("a","b","c")
z <- data.frame(a,b)
table(z$a,z$b)
with(z,{table(a,b)
  summary(z)})
with(z,{p <<- z+1
})
#因子
x <- c("Type","Type1","Type")
x <- factor(x)
x <- factor(x,ordered = T)
x <- factor(x,ordered = T,levels = c("Type1","Type"))
sex <- c(1,1,1,2,2,1)
sex <- factor(sex,levels = c(1,2),labels = c("man","woman"))
#列表
a <- "my first list"
b <- c(1,2,3,4)
c <- matrix(1:24,4,6,byrow = T)
d <- c("one","two")
e <- list(A=a,B=b,C=c,D=d)
e[2]
e[[2]]
#键盘输入
mydata <- data.frame(a=numeric(0),b=character(0),c=numeric(0))
mydata <- edit(mydata)
fix(mydata)
mydatatxt <- "
age gender weight
25 m 166
30 f 115
18 f 120"
mydata <- read.table(header = T,text = mydatatxt)
#文件输入
x <- read.xlsx("2015年度中国城市GDP排名.xlsx",1)
#还原par
opar <- par(no.readonly = TRUE)
par(opar)
#自行添加坐标轴
plot(ann=FALSE)
axis(at=)
#保存图片至pdf
pdf("mygraph.pdf")
attach(mtcars)
plot(wt,mpg)
abline(lm(mpg~wt))
dev.off()
#同时画多幅图但不覆盖
dev.new()#打开一个新窗口，也可在之后再打开一个
attach(mtcars)
plot(wt,mpg)
abline(lm(mpg~wt))
attach(mtcars)
plot(wt,mpg)
#改变字体
windowsFonts(A="Arial Black")
par(family="A")
#添加文本、自定义坐标轴和图例
dose <- seq(20,60,10)
drugA <- c(16,20,27,40,60)
drugB <- c(15,18,25,31,40)
plot(dose,drugA,type="b",col="red",lty=2,pch=2,lwd=2,xlim = c(0,60),ylim = c(0,65),ann = FALSE)
title(main="A",sub="B",xlab="D",ylab="D",col.lab="green",cex.lab=0.75)
#加自定义坐标轴
x <- c(1:10)
y <- x
z <- 10/x
opar <- par(no.readonly = TRUE)
plot(x,y,type="b")
lines(x,z,type="b")
axis(2,at=x,labels=x)
axis(4,at=z,labels=round(z,digits = 2))
mytext("y=1/x",side=4)
#加图例
x <- c(1:10)
y <- x
z <- 10/x
opar <- par(no.readonly = TRUE)
plot(x,y,type="b")
lines(x,z,type="b")
legend("topleft",inset=.05,title="A",c("A","b"),lty = 1,pch = 15)
#加文本
attach(mtcars)
plot(wt,mpg)
text(wt,mpg,row.names(mtcars),pos=4)
mtext("a",side=4)
#多幅图排布
attach(mtcars)
par(mfrow=c(3,1))
layout(matrix(c(1,1,2,3),2,2,byrow=TRUE),widths = c(3,1),heights = c(2,1))
hist(wt)
hist(wt)
hist(wt)
#精细排布
attach(mtcars)
par(fig=c(0,0.8,0,0.8))#主图横纵范围为0-0.8
plot(wt,mpg)
par(fig=c(0.65,1,0,0.8),new=TRUE)#横纵范围0.65-0.1和0-0.8
boxplot(wt,horizontal = TRUE,axes=FALSE)#axes为画轴否
#transform数据
a <- data.frame(x1=c(1:5),x2=seq(1,10,2))
b <- transform(a,x3=x1+x2,x4=(x1+x2)/2)
#within直接操作数据
a <- data.frame(x1=c(1:5),x2=seq(1,10,2))
a <- within(a,{x3 <- NA
x3[x1>3] <- "A"})#within可修改数据框
#缺失值处理
x <- c(1:3,NA)
y <- sum(x,na.rm = TRUE)#na.rm
x <- as.data.frame(x)
x <- na.omit(x)#na.omit行删除
#日期值
x <- as.Date(c("2006-05-08"))
x <- as.Date(c("2006,05,08"),"%Y,%m,%d")
x <- format(Sys.Date(),format="%A")
#数据排序
x <- with(mtcars,mtcars[order(cyl,wt),])
#合并数据集
x <- merge(dataframe.A,dataframe.B,by=c("ID"))
x <- cbind(dataframe.A,dataframe.B)#合并行，若B中有不存在的变量则设为NA
#丢弃变量
x <- mtcars[c(-1,-2)]#1
unusefulnames <- names(mtcars)%in%c("disp")#2
x <- mtcars[!unusefulnames]
x <- mtcars#3
x$mpg <- x$cyl <- NULL
#subset选择观测函数
x <- subset(mtcars,qsec<17,wt:carb)
#随机抽样
x <- mtcars[sample(1:nrow(mtcars),3,replace = FALSE),]
#数据中心化和标准化处理
x <- c(19,21,23)
y <- scale(x,center = TRUE,scale = TRUE)
#设置随机数种子
set.weed(2)#随机数重现
runif(n)
#多元正态分布数据
n=500
mean=c(a,b,c)
sigma <- matrix(c(),nrow=3,ncol=3)
mvrnorm(n,mean,sigma)
#数据处理实例
options(digits=2)
students <- c("bwb","sqr","gyh","sjh","lh","zzr","yzh")
math <- c(67,98,76,85,79,87,78)
science <- c(76,78,82,89,85,91,90)
physics <- c(73,98,76,78,67,89,65)
x <- data.frame(students,math,science,physics)
y <- scale(x[,-1])
x$score <- apply(y,1,mean)
z <- quantile(x$score,0.5)
x$grade[x$score<z] <- "B"
x$grade[x$score>=z] <- "A"
x <- x[order(x$students),]
#循环
for(i in 1:10) print("Hello")
i <- 10
while(i>0)
{print("hello")
  i <- i-1}
#条件
if(is.character("abc")){
  print("bullish")
}
ifelse(6<5,print(1),print(2))
print(switch("A",A="aoe",B="aoee"))
#编写函数
mydate <- function(type="long"){
  switch(type,
         long=format(Sys.time(),"%A %B %d %Y"),
         short=format(Sys.time(),"%m-%d-%y"),
         cat(type,"is not correct"))
}
#整合数据
attach(mtcars)
mtcars1 <- aggregate(mtcars,by=list(gear),mean,na.rm=TRUE)
#reshape2
library(reshape2)
x <- as.data.frame(Titanic)
x <- melt(x,id=c("Sex","Survived"))
x <- x[65:96,]
x$value <- as.numeric(x$value)
x <- dcast(x,Sex~Survived,sum)
row.names(x)=x$Sex
x <- x[,-1]
x <- as.matrix(x)
#条形图
library(vcd)#数据集
plot(Arthritis$Improved,main="Bar Plot",xlab="Improvement",ylab="Frequency")
counts <- table(Arthritis$Improved)
barplot(counts,main="Bar Plot",xlab="Improvement",ylab="Frequency")
#堆砌和分组条形图
library(vcd)
counts <- table(Arthritis$Improved,Arthritis$Treatment)
barplot(counts,xlab="Improvement",ylab="Frequency",legend=rownames(counts))#beside=TRUE分组条形图
#均值条形图
states <- data.frame(state.region,state.x77)
means <- aggregate(states$Illiteracy,by=list(state.region),mean)
barplot(means$x,names.arg=means$Group.1)
title("Mean Illiteracy Rate")
#棘状图
library(vcd)
attach(Arthritis)
counts <- table(Treatment,Improved)
spine(counts,main="Spinogram Example")
#饼图
slices <- c(10,9,4,2,4,5)
lbls <- paste(c("a","b","c","d","e","f")," ",round(slices/sum(slices)*100),"%")
pie(slices,labels = lbls,main="X",col = rainbow(6))
library(plotrix)
fan.plot(slices,labels = lbls,main="X",col = rainbow(6))#扇形图
#直方图
hist(mtcars$mpg,breaks = 12,col="red",freq = FALSE)
rug(jitter(mtcars$mpg))#描述点的区间分布
lines(density(mtcars$mpg),col="blue",lwd=2)
box()
#核密度图
plot(density(mtcars$mpg))
polygon(density(mtcars$mpg),col="red",border="blue")#填充颜色
rug(jitter(mtcars$mpg),col="brown")
#可比较的核密度图
library(sm)
attach(mtcars)
factor <- factor(cyl,levels = c(4,6,8),labels = c("4 cyclinder","6 cyclinder","8 cyclinder"))
sm.density.compare(mpg,cyl,xlabs="X")
title(main="abcde")
legend(locator(1),levels(factor),fill=4)
#箱线图
attach(mtcars)
boxplot(mpg~cyl,main="box plot",xlab="X",ylab="Y",notch=TRUE,col="red")
mtcars$cyl.f <- factor(mtcars$cyl,levels = c(4,6,8),labels = c("4","6","8"))
mtcars$am.f <- factor(mtcars$am,levels = c(0,1),labels = c("auto","standard"))
boxplot(mpg~am.f*cyl.f)
library(vioplot)#小提琴图
x1 <- mtcars$mpg[mtcars$cyl==4]
x2 <- mtcars$mpg[mtcars$cyl==6]
x3 <- mtcars$mpg[mtcars$cyl==8]
vioplot(x1,x2,x3)
#点图
dotchart(mtcars$mpg,labels = row.names(mtcars))
x <- mtcars[order(mtcars$mpg),]
x$cyl.f <- factor(x$cyl)
x$cyl <- as.numeric(x$cyl)
for(i in 1:length(rownames(x))){x$color[i] <- switch(x$cyl[i]/2-1,"red","blue","black")}
dotchart(x$mpg,labels = row.names(x),cex=1,groups = x$cyl,gcolor="black",pch=19,color = x$color)
#summary
summary(mtcars)
#sapply
sapply(mtcars,mean,na.omit=TRUE)
#Hmisc
library(Hmisc)
describe(mtcars)
#stat.desc
library(pastecs)
stat.desc(mtcars,norm=TRUE,p=0.95)
#psych
library(psych)
describe(mtcars)
Hmisc::describe(mtcars)
#分组描述
aggregate(mtcars,by=list(am=mtcars$am),mean)
by(mtcars[c("mpg","hp","wt")],mtcars$am,describe)
library(doBy)
summaryBy(mpg+wt+hp~am,data=mtcars,FUN=stat.desc)
library(psych)
describeBy(mtcars[c("mpg","wt","hp")],list(am=mtcars$am))
#频数表和列联表
x <- with(Arthritis,table(Improved))
prop.table(x)
y <- with(Arthritis,table(Improved,Treatment))
#二维列联表
y <- xtabs(~Treatment+Improved,data=Arthritis)
margin.table(y,2)
prop.table(y,1)
addmargins(y,1)
library(gmodels)
CrossTable(Arthritis$Treatment,Arthritis$Improved)
#多维列联表
x <- with(Arthritis,table(Sex,Improved,Treatment))
x <- xtabs(~Sex+Improved+Treatment,data=Arthritis)
y <- ftable(x)
#卡方独立检验
options(digits = 6)
library(vcd)
x <- xtabs(~Treatment+Improved,data = Arthritis)
chisq.test(x)
#fisher精确检验
x <- xtabs(~Treatment+Improved,data=Arthritis)
fisher.test(x)
#mantelhaen.test
x <- xtabs(~Treatment+Improved+Sex,data=Arthritis)
mantelhaen.test(x)
#相关性的度量
library(vcd)
x <- xtabs(~Treatment+Improved,data=Arthritis)
assocstats(x)
#相关系数
cov(state.x77)#协方差
cor(state.x77,use="all.obs",method="pearson")
x <- state.x77[,c("Population","Income","Illiteracy")]
y <- state.x77[,c("Frost","Area")]
cor(x,y)
cor.test(state.x77[,2],state.x77[,3],alternative="less",method="pearson")
corr.test(state.x77,use = "complete")
library(psych)
r.test(30,0.99)
#偏相关
library(ggm)
pcor <- pcor(c(1,5,3,4,2),cov(state.x77))
pcor.test(pcor,3,49)
#独立样本t检验
library(MASS)
t.test(Prob~So,data=UScrime)
#非独立样本t检验
with(UScrime,t.test(U1,U2,paired=TRUE))
#Wilcoxon秩和检验
wilcox.test(Prob~So,alternative="less",data=UScrime)
with(UScrime,wilcox.test(U1,U2,paired=TRUE))
#多于两组的非参数检验
states <- data.frame(state.region,state.x77)
kruskal.test(Illiteracy~state.region,data=states)#独立
#wmc函数
source("http://www.statmethods.net/RiA/wmc.txt")
states <- data.frame(state.region,state.x77)
wmc(Illiteracy~state.region,data=states,method = "holm")
#lm函数
fit <- lm(height~weight,data=women)
summary(fit)
fitted(fit)
residuals(fit)
plot(women$weight,women$height)
abline(fit)
#多项式回归
fit <- lm(weight~height+I(height^2),data=women)
summary(fit)
plot(women$height,women$weight)
lines(women$height,fitted(fit))
#多元线性回归
states <- as.data.frame(state.x77[,c(5,1,3,2,7)])
cor(states)
library(car)
scatterplotMatrix(states,spread=FALSE,smoother.args=list(lty=2))
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
summary(fit)
#交互项多元线性回归
fit <- lm(mpg~hp+wt+hp:wt,data=mtcars)
summary(fit)
fit <- lm(mpg~(hp+wt)^2,data=mtcars)
library(effects)
plot(effect("hp:wt",fit,,list(wt=c(2.2,3.2,4.2))),multiline=TRUE)
#回归诊断
options(digits=3)
states <- as.data.frame(state.x77[,c(5,1,3,2,7)])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
confint(fit)
#检验回归分析
fit <- lm(weight~height,data=women)
par(mfrow=c(2,2))
plot(fit)
#car包检测残差合理性
library(car)
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
qqPlot(fit,labels=row.names(states),id=list(method="identify"),simulate = TRUE,main="Q-Q Plot")
crPlots(fit)
vif(fit)#sqrt(vif>2)有多重共线性问题
#学生化残差图
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
residplot <- function(fit,nbreaks=10){
  z <- rstudent(fit)
  hist(z,breaks=nbreaks,freq=FALSE)
  rug(jitter(z),col="brown")
  curve(dnorm(x,mean=mean(z),sd=sd(z)),add=TRUE,col="blue",lwd=2)
  lines(density(z)$x,density(z)$y,col="red",lwd=2,lty=2)
  legend("topright",legend=c("a","b"),lty=1:2,col=c("blue","red"),cex=.7)
}
residplot(fit)
#误差的独立性
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
library(car)
durbinWatsonTest(fit)
#同方差性
library(car)
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
ncvTest(fit)
spreadLevelPlot(fit)#水平
#线性模型假设的综合验证
library(gvlma)
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
ncvTest(fit)
gvmodel <- gvlma(fit)
summary(gvmodel)
#离群点统计
library(car)
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
outlierTest(fit)#只显示单个最大残差值是否显著
#高杠杆值点统计
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
hat.plot <- function(fit){
  p <- length(coefficients(fit))
  n <- length(fitted(fit))
  plot(hatvalues(fit))
  abline(h=c(2,3)*p/n,col="red",lty=2)
  identify(1:n,hatvalues(fit),names(hatvalues(fit)))
}
hat.plot(fit)
#强影响点统计
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
cutoff <- 4/(nrow(states)-length(coefficients(fit))-1)
plot(fit,which=4)
abline(h=cutoff,lty=2,col="red")
library(car)
avPlots(fit,ask=FALSE,id=list(method="identify"))
influencePlot(fit,id=list(method="identify"))
#变量变换
library(car)
summary(powerTransform(states$Murder))#是否需要正态化变量
boxTidwell(Murder~Population+Illiteracy,data=states)#合适的lambda
#回归模型的选择
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit1 <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
fit2 <- lm(Murder~Population+Illiteracy,data=states)
anova(fit1,fit2)
AIC(fit1,fit2)
#变量选择
library(MASS)
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit1 <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
stepAIC(fit,direction = "backward")
library(leaps)
leaps <- regsubsets(Murder~Population+Illiteracy+Income+Frost,data=states,nbest = 6)#nbest为每个变量数量的子集量
plot(leaps,scale="adjr2")
library(car)
subsets(leaps,statistic="cp")
abline(a=1,b=1,lty=2,col="red")
#交叉验证
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
shrinkage <- function(fit,k=10){
  require(bootstrap)
  
  theta.fit <- function(x,y){lsfit(x,y)}
  theta.predict <- function(fit,x){cbind(1,x)%*%fit$coefficients}
  
  x <- fit$model[,2:ncol(fit$model)]
  y <- fit$model[,1]
  results <- crossval(x,y,theta.fit,theta.predict,ngroup=k)
  r <- cor(y,fit$fitted.values)^2
  rcv <- cor(y,results$cv.fit)^2
  cat("Original R-square =",r,"\n")
  cat(k,"重交叉验证 R-square =",rcv,"\n")
  cat("Change =",rcv-r,"\n")
}
shrinkage(fit)
fit2 <- lm(Murder~Population+Illiteracy,data=states)
shrinkage(fit2)
#标准化回归系数
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
sstates <- as.data.frame(scale(states))
sfit <- lm(Murder~Population+Illiteracy+Income+Frost,data=sstates)
coef(sfit)
#预测变量相对权重
relweights <- function(fit,...){
  R <- cor(fit$model)
  nvar <- ncol(R)    
  rxx <- R[2:nvar, 2:nvar]    
  rxy <- R[2:nvar, 1]    
  svd <- eigen(rxx)    
  evec <- svd$vectors    
  ev <- svd$values    
  delta <- diag(sqrt(ev))    
  lambda <- evec %*% delta %*% t(evec)    
  lambdasq <- lambda ^ 2    
  beta <- solve(lambda) %*% rxy    
  rsquare <- colSums(beta ^ 2)    
  rawwgt <- lambdasq %*% beta ^ 2
  import <- (rawwgt / rsquare) * 100
  import <- as.data.frame(import)    
  row.names(import) <- names(fit$model[2:nvar])    
  names(import) <- "Weights"
  import <- import[order(import),1, drop=FALSE]    
  dotchart(import$Weights, labels=row.names(import),           
           xlab="% of R-Square", pch=19,           
           main="Relative Importance of Predictor Variables",           
           sub=paste("Total R-Square=", round(rsquare, digits=3)),           
           ...)
  return(import)
}
states <- as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
fit <- lm(Murder~Population+Illiteracy+Income+Frost,data=states)
relweights(fit,col="blue")
#单因素方差分析
library(multcomp)
attach(cholesterol)
table(trt)
aggregate(response,by=list(trt),mean)
fit <- aov(response~trt)
library(gplots)
plotmeans(response~trt)
#多重比较
library(multcomp)
attach(cholesterol)
fit <- aov(response~trt)
TukeyHSD(fit)
par(las=1)
par(mar=c(5,8,4,2))
plot(TukeyHSD(fit))
par(mar=c(5,4,6,2))
tuk <- glht(fit,linfct=mcp(trt="Tukey"))
plot(cld(tuk,level=0.05),col="red")#相同字母说明并不显著
#评估检验的假设条件（正态分布等）
library(multcomp)
library(car)
qqPlot((lm(response~trt,data=cholesterol)),stimulate=TRUE)
bartlett.test(response~trt,data=cholesterol)
attach(cholesterol)
fit <- aov(response~trt)
outlierTest(fit)
#单因素协方差分析
data(litter,package = "multcomp")
attach(litter)
table(litter$dose)
aggregate(weight,by=list(litter$dose),mean)
fit <- aov(weight~gesttime+dose,data=litter)
library(effects)
effect("dose",fit)
library(multcomp)
#多重比较
library(multcomp)
fit <- aov(weight~gesttime+dose,data=litter)
contrast <- rbind("no drug vs. drugs"=c(100,1,1,1))#第一组和其他三组比较
summary(glht(fit,linfct = mcp(dose=contrast)))
#评估检验地假设条件
library(multcomp)
fit <- aov(weight~gesttime*dose,data=litter)#gesttime作为协变量条件成立
summary(fit)
#分析结果可视化
library(HH)
ancova(weight~gesttime+dose,data=litter)#截距同质性
#双因素方差分析
attach(ToothGrowth)
table(supp,ToothGrowth$dose)
aggregate(len,by=list(ToothGrowth$dose,supp),mean)
ToothGrowth$dose <- factor(ToothGrowth$dose)
fit <- aov(len~supp*ToothGrowth$dose)
summary(fit)
interaction.plot(ToothGrowth$dose,supp,len,type="b")
library(gplots)
plotmeans(len~interaction(supp,ToothGrowth$dose,sep=" "),connect = list(c(1,3,5),c(2,4,6)))
library(HH)
interaction2wt(len~supp*ToothGrowth$dose)
#重复测量方差分析
CO2$conc <- factor(CO2$conc)
x <- subset(CO2,Treatment=="chilled")
fit <- aov(uptake~conc*Type+Error(Plant/(conc)),x)
summary(fit)
par(las=2,mar=c(10,4,4,2))
with(x,interaction.plot(conc,Type,uptake,type="b"))
boxplot(uptake~Type*conc,data=x)
#多元方差分析
library(MASS)
attach(UScereal)
shelf <- factor(shelf)
y <- cbind(calories,fat,sugars)
aggregate(y,by=list(shelf),mean)
cov(y)
fit <- manova(y~shelf)
summary(fit)
summary.aov(fit)
#评估假设检验
library(MASS)
attach(UScereal)
shelf <- factor(shelf)
y <- cbind(calories,fat,sugars)
z <- colMeans(y)
a <- mahalanobis(y,z,cov(y))
p <- qqplot(qchisq(ppoints(nrow(y)),df=ncol(y)),a)
abline(a=0,b=1)
identify(p$x,p$y,labels = rownames(UScereal))
library(mvoutlier)
outliers <- aq.plot(y)
#稳健多元方差分析
library(MASS)
attach(UScereal)
shelf <- factor(shelf)
y <- cbind(calories,fat,sugars)
library(rrcov)
Wilks.test(y,shelf,method="mcd")
#lm回归进行方差分析
library(multcomp)
fit <- lm(response~trt,data=cholesterol)
summary(fit)
options(contrasts = c("contr.SAS","contr.helmert"))
fit <- lm(response~trt,data=cholesterol)
summary(fit)
#t检验功效分析
library(pwr)
pwr.t.test(d=.8,sig.level=.05,power = .9,type = "two.sample",alternative = "two.sided")
pwr.t.test(n=20,d=.5,,sig.level = .01,type = "two.sample",alternative = "two.sided")
pwr.t2n.test(n1=20,n2=25,d=1,sig.level = .01,alternative = "two.sided")#pwr.t2n.test两个样本量不同
#方差功效分析
library(pwr)
pwr.anova.test(k=5,f=.25,sig.level = .05,power = .8)
#相关性功效分析
library(pwr)
pwr.r.test(r=.25,sig.level = .05,power = .90,alternative = "greater")
#线性模型功效分析
library(pwr)
pwr.f2.test(u=3,f2=0.0769,sig.level = .05,power=.9)
#比例检验
library(pwr)
pwr.2p.test(h=ES.h(.65,.6),sig.level = .05,power = .9,alternative = "greater")
#卡方检验功效分析
library(pwr)
prob <- matrix(c(.42,.28,.03,.07,.10,.10),byrow = TRUE,nrow=3)
pwr.chisq.test(w=ES.w2(prob),df=2,sig.level = .05,power = .9)
#选择合适的效应值
library(pwr)
es <- seq(.1,.5,.01)
nes <- length(es)
samsize <- NULL
for(i in 1:nes){
  result <- pwr.anova.test(k=5,f=es[i],sig.level = .05,power = .9)
  samsize[i] <- ceiling(result$n)
}
plot(samsize,type="l",lwd=2,col="red")
#绘制功效分析图形
library(pwr)
r <- seq(.1,.5,.05)
nr <- length(r)
p <- seq(.4,.9,.1)
np <- length(p)
samsize <- array(numeric(nr*np),dim = c(nr,np))
for(i in 1:np){
  for(j in 1:nr){
    result <- pwr.r.test(n=NULL,r=r[j],sig.level = .05,power=p[i],alternative = "two.sided")
    samsize[j,i] <- ceiling((result$n))
  }
}
xrange <- range(r)
yrange <- round(range(samsize))
colors <- rainbow(length(p))
plot(xrange,yrange,type="n")
for(i in 1:np){
  lines(r,samsize[,i],type="l",lwd=2,col=colors[i])
}