NAME="sin1248_to_polynominal"
python multiple_burgers.py --niter=50000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_polynominal.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_sin_divide2.mat" "../MyData/burgers_sin_divide1.mat" "../MyData/burgers_sine.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos1248_to_polynominal"
python multiple_burgers.py --niter=50000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_polynominal.mat" --traindata "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_cos_divide2.mat" "../MyData/burgers_cos_divide1.mat" "../MyData/burgers_cos.mat" && git add . && git commit -m $NAME && git push origin master
NAME="sin1248_cos1248_to_polynominal"
python multiple_burgers.py --niter=50000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_polynominal.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_sine.mat" "../MyData/burgers_sin_divide2.mat" "../MyData/burgers_sin_divide1.mat" "../MyData/burgers_cos.mat" "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_cos_divide2.mat" "../MyData/burgers_cos_divide1.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_polynominal_to_polynominal"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME --testdata="../MyData/burgers_polynominal.mat" --traindata "../MyData/burgers_cos_polynominal.mat" && git add . && git commit -m $NAME && git push origin master
NAME="sine_divide4_to_polynominal"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_polynominal.mat" --traindata "../MyData/burgers_sin_divide4.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_divide4_to_polynominal"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_polynominal.mat" --traindata "../MyData/burgers_cos_divide4.mat" && git add . && git commit -m $NAME && git push origin master
NAME="sine_divide4_sine_to_polynominal"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_polynominal.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_sine.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_divide4_cos_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_cos.mat" && git add . && git commit -m $NAME && git push origin master
NAME="sine_divide4_cos_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_cos.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_divide4_sine_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_sine.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_divide4_sine_divide4_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_cos_divide4.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_divide4_cos_sine_divide4_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_cos.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_divide4_cos_sine_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_sine.mat" "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_cos.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_sine_sine_divide4_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_cos.mat" "../MyData/burgers_sine.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_divide4_sine_divide4_sine_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_sine.mat" && git add . && git commit -m $NAME && git push origin master
NAME="cos_sine_cos_divide4_sin_divide4_to_square"
python multiple_burgers.py --niter=30000 --scipyopt=True --name=$NAME  --testdata="../MyData/burgers_square.mat" --traindata "../MyData/burgers_sin_divide4.mat" "../MyData/burgers_cos_divide4.mat" "../MyData/burgers_sine.mat" "../MyData/burgers_cos.mat" && git add . && git commit -m $NAME && git push origin master
