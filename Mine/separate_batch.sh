NAME="sine_to_exp_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../Data/burgers.mat" --traindata="../MyData/burgers_sine.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="cos_to_exp_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../Data/burgers.mat" --traindata="../MyData/burgers_cos.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="cos_sin_to_exp_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../Data/burgers.mat" --traindata "../MyData/burgers_cos.mat" "../MyData/burgers_sine.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="exp_to_sine_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_sine.mat" --traindata "../Data/burgers.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="cos_to_sine_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_sine.mat" --traindata "../MyData/burgers_cos.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="exp_cos_to_sine_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_sine.mat" --traindata "../Data/burgers.mat" "../MyData/burgers_cos.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="exp_to_cos_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_cos.mat" --traindata "../Data/burgers.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="sine_to_cos_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_cos.mat" --traindata "../MyData/burgers_sine.mat"
done
git add . && git commit -m $NAME && git push origin master
NAME="exp_sine_to_cos_separate"
for i in `seq 10`
do
python multiple_burgers_separate.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_cos.mat" --traindata "../Data/burgers.mat" "../MyData/burgers_sine.mat"
done
git add . && git commit -m $NAME && git push origin master
