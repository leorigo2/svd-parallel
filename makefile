svd_power: svd_power.c
	gcc svd_power.c -lm -o svd_power

clean:
	rm -f svd_power