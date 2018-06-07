# CUDAカーネル内でのfor文の実行速度に関する実験
## 概要
- forを回す回数(N)はconst intで指定するのとtemplate引数で指定するので差があるか
- 明示的なpragma unrollによる影響はあるか
- `for(i:N){func(i)}` と `for(i:N/2){func(2*i);func(2*i+1)}` で実行速度に差はあるか

## 実験プログラム
- カーネル  
配列の全要素に-1.0fをかける
```cpp
__global__ void dev_kernel_A(const int N,float *ptr){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= N) return;

	for(int i = 0;i < N;i++){
		ptr[i] = ptr[i] * (-1.0f);
	}
}
```

## 結果

- GF GTX 1080


```
##-- ROUND 1
#-- simple for
ptr : 0x7f6fb7400000
Elapsed time : 94946 [us]
#-- divided by 2 for
ptr : 0x7f6fb7400000
Elapsed time : 51706 [us]
#-- counter assembled simple for
ptr : 0x7f6fb7400000
Elapsed time : 79729 [us]
#-- counter assembled divided by 2 for
ptr : 0x7f6fb7400000
Elapsed time : 36507 [us]
#-- counter assembled simple for (expressly unrolled)
ptr : 0x7f6fb7400000
Elapsed time : 76778 [us]
#-- counter assembled divided by 2 for (expressly unrolled)
ptr : 0x7f6fb7400000
Elapsed time : 33598 [us]

##-- ROUND 2
#-- simple for
ptr : 0x7f6fb7400000
Elapsed time : 80372 [us]
#-- divided by 2 for
ptr : 0x7f6fb7400000
Elapsed time : 44132 [us]
#-- counter assembled simple for
ptr : 0x7f6fb7400000
Elapsed time : 73265 [us]
#-- counter assembled divided by 2 for
ptr : 0x7f6fb7400000
Elapsed time : 33454 [us]
#-- counter assembled simple for (expressly unrolled)
ptr : 0x7f6fb7400000
Elapsed time : 73293 [us]
#-- counter assembled divided by 2 for (expressly unrolled)
ptr : 0x7f6fb7400000
Elapsed time : 33423 [us]

##-- ROUND 3
#-- simple for
ptr : 0x7f6fb7400000
Elapsed time : 80353 [us]
#-- divided by 2 for
ptr : 0x7f6fb7400000
Elapsed time : 44147 [us]
#-- counter assembled simple for
ptr : 0x7f6fb7400000
Elapsed time : 73164 [us]
#-- counter assembled divided by 2 for
ptr : 0x7f6fb7400000
Elapsed time : 33753 [us]
#-- counter assembled simple for (expressly unrolled)
ptr : 0x7f6fb7400000
Elapsed time : 73292 [us]
#-- counter assembled divided by 2 for (expressly unrolled)
ptr : 0x7f6fb7400000
Elapsed time : 33666 [us]
```
