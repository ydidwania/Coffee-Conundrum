#!/bin/bash

for ins in "../instances/i-1.txt" "../instances/i-2.txt" "../instances/i-3.txt"; do
	for rs in {0..49}; do
		for hz in 50 200 800 3200 12800 51200 204800; do
			for algo in "round-robin" "ucb" "kl-ucb" "thompson-sampling"; do
				python algo.py --instance $ins --algorithm $algo --randomSeed $rs --epsilon 0.002 --horizon $hz >> output.txt
			done
			for eps in 0.002 0.02 0.2; do
				python algo.py --instance $ins --algorithm epsilon-greedy --randomSeed $rs --epsilon $eps --horizon $hz >> output.txt
			done
		done
	done
done 


