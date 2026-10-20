import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# 配置参数（使用你当前的设置）
min_mz = 0.005
max_mz = 6000.0
num_freqs = 2048
fourier_power = 3.0

indices = np.arange(num_freqs)
x = indices / (num_freqs - 1)          # 归一化索引
t = x ** fourier_power                 # power 变换
freqs = (1 / min_mz) * ((max_mz / min_mz) ** (-t))  # 频率
mass_diff = 1 / freqs                  # 对应的质量差（分辨率）

# 创建双 y 轴图
fig, ax1 = plt.subplots(figsize=(8, 5))

# 左轴：频率
ax1.plot(x, freqs, 'b-', linewidth=2, label='频率 (1/Da)')
ax1.set_xlabel('归一化索引 (i/(N-1))')
ax1.set_ylabel('频率 (1/Da)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('log')  # 频率跨度大，使用对数刻度
ax1.grid(True, linestyle=':', alpha=0.7)

# 右轴：质量差
ax2 = ax1.twinx()
ax2.plot(x, mass_diff, 'r-', linewidth=2, label='可分辨质量差 (Da)')
ax2.set_ylabel('可分辨质量差 (Da)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_yscale('log')  # 同样对数刻度

# 添加参考线：标记 1 Da 和 0.036 Da (K/Q 差)
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(y=0.036, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.text(0.05, 1.2, '1 Da', color='gray')
ax2.text(0.05, 0.04, '0.036 Da (K/Q)', color='gray')

plt.title(f'频率分布 (power={fourier_power}, N={num_freqs})')
fig.tight_layout()
plt.show()
