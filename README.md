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

### 動作環境
- C++ >= c++14 (11でもいいかもだけどMakefileを書き換えてね)
- CUDA >= (人権)

## 結果

- GF GTX 1080


```
array size (N) : 1000
number of "for" computation (C) : 1000
ROUND : 3

##-- ROUND 1
#-- simple for
ptr : 0x7fb517400000
Elapsed time : 94335 [us]
#-- divided by 2 for
ptr : 0x7fb517400000
Elapsed time : 48053 [us]
#-- N assembled, simple for
ptr : 0x7fb517400000
Elapsed time : 78853 [us]
#-- N assembled, divided by 2 for
ptr : 0x7fb517400000
Elapsed time : 36426 [us]
#-- N assembled, simple, for (expressly unrolled)
ptr : 0x7fb517400000
Elapsed time : 74075 [us]
#-- N assembled, divided by 2, for (expressly unrolled)
ptr : 0x7fb517400000
Elapsed time : 33764 [us]

##-- ROUND 2
#-- simple for
ptr : 0x7fb517400000
Elapsed time : 81008 [us]
#-- divided by 2 for
ptr : 0x7fb517400000
Elapsed time : 44825 [us]
#-- N assembled, simple for
ptr : 0x7fb517400000
Elapsed time : 73965 [us]
#-- N assembled, divided by 2 for
ptr : 0x7fb517400000
Elapsed time : 33696 [us]
#-- N assembled, simple, for (expressly unrolled)
ptr : 0x7fb517400000
Elapsed time : 74229 [us]
#-- N assembled, divided by 2, for (expressly unrolled)
ptr : 0x7fb517400000
Elapsed time : 33869 [us]

##-- ROUND 3
#-- simple for
ptr : 0x7fb517400000
Elapsed time : 81048 [us]
#-- divided by 2 for
ptr : 0x7fb517400000
Elapsed time : 44849 [us]
#-- N assembled, simple for
ptr : 0x7fb517400000
Elapsed time : 74117 [us]
#-- N assembled, divided by 2 for
ptr : 0x7fb517400000
Elapsed time : 33730 [us]
#-- N assembled, simple, for (expressly unrolled)
ptr : 0x7fb517400000
Elapsed time : 74110 [us]
#-- N assembled, divided by 2, for (expressly unrolled)
ptr : 0x7fb517400000
Elapsed time : 33786 [us]
```
