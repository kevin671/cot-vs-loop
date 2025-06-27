# 入出力がnだからやりやすいか？
# nを増やして必要なループ数の増加を調べる

# fftの本質ってなんだろう

# The Number Theoretic Transform (NTT) is a mathematical transformation similar to the Discrete Fourier Transform (DFT), but it operates within a finite ring, making it suitable for efficient polynomial multiplication and convolution in various applications, particularly in cryptography and digital signal processing.

import numpy as np


def modinv(x, p):
    """x の逆元を法 p で求める"""
    return pow(x, -1, p)


def ntt(a, root, p):
    """Number Theoretic Transform"""
    n = len(a)
    levels = n.bit_length() - 1
    a = list(a)

    # ビット反転置換
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Cooley-Tukey
    length = 2
    while length <= n:
        wlen = pow(root, n // length, p)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w % p
                a[i + j] = (u + v) % p
                a[i + j + length // 2] = (u - v) % p
                w = w * wlen % p
        length *= 2
    return a


def generate_ntt_dataset(num_samples=10000, length=16, p=7340033, g=3):
    """
    - num_samples: サンプル数
    - length: 入力長さ（必ず 2^k）
    - p: 素数（p ≡ 1 mod length）
    - g: 原始根
    """
    root = pow(g, (p - 1) // length, p)  # n次単位根
    inputs, outputs = [], []

    for _ in range(num_samples):
        a = np.random.randint(0, p, size=length, dtype=np.int64)
        a_ntt = ntt(a, root, p)
        inputs.append(a)
        outputs.append(np.array(a_ntt, dtype=np.int64))

    return np.stack(inputs), np.stack(outputs), p


# p = 17, n = 16
# p = 193, n = 64
# できるのかな...
if __name__ == "__main__":
    # Example usage
    inputs, outputs, p = generate_ntt_dataset(num_samples=10, length=16)
    print("Inputs:")
    print(inputs)
    print("Outputs:")
    print(outputs)
    print("Modulus p:", p)
