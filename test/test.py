import numpy as np
import unittest


class MyTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(1 + 2, 3)

    def test_sub(self):
        self.assertEqual(3 - 2, 1)


class TestTensorBroadcasting(unittest.TestCase):
    """测试张量广播机制"""

    def test_basic_broadcasting(self):
        """测试基本广播：二维数组加一维数组"""
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])  # shape: (2, 3)

        B = np.array([10, 20, 30])  # shape: (3,)

        # 手动计算期望结果
        expected = np.array([[1 + 10, 2 + 20, 3 + 30],
                             [4 + 10, 5 + 20, 6 + 30]])

        # 使用广播相加
        C = A + B

        # 验证形状
        self.assertEqual(C.shape, (2, 3), "广播后形状应为 (2, 3)")

        # 验证数值
        np.testing.assert_array_equal(C, expected, "广播计算结果不正确")

        # 验证每个元素的计算方式
        for i in range(2):
            for j in range(3):
                self.assertEqual(C[i, j], A[i, j] + B[j],
                                 f"元素({i},{j})计算错误: {C[i, j]} != {A[i, j]} + {B[j]}")

    def test_broadcast_dimensions(self):
        """测试广播的维度匹配规则"""
        # 测试从右边开始对齐
        A = np.ones((3, 4, 5))  # shape: (3, 4, 5)
        B = np.ones((4, 5))  # shape: (4, 5) → 广播为 (1, 4, 5) → (3, 4, 5)

        C = A + B
        self.assertEqual(C.shape, (3, 4, 5))

        # 测试形状为 (1, n) 的广播
        D = np.array([[1, 2, 3]])  # shape: (1, 3)
        E = np.array([[4], [5]])  # shape: (2, 1)
        F = D + E  # 广播为 (2, 3)

        expected_F = np.array([[1 + 4, 2 + 4, 3 + 4],
                               [1 + 5, 2 + 5, 3 + 5]])
        np.testing.assert_array_equal(F, expected_F)

    def test_invalid_broadcast(self):
        """测试无效的广播（应该抛出异常）"""
        A = np.ones((2, 3))
        B = np.ones((4,))  # shape: (4,) 无法广播到 (2, 3)

        with self.assertRaises(ValueError):
            C = A + B

    def test_broadcast_with_scalar(self):
        """测试标量广播"""
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])  # shape: (2, 3)

        # 标量广播到整个数组
        C = A + 10

        expected = np.array([[11, 12, 13],
                             [14, 15, 16]])
        np.testing.assert_array_equal(C, expected)

    def test_3d_broadcasting(self):
        """测试三维张量广播"""
        # 形状: (2, 1, 3) 广播到 (2, 4, 3)
        A = np.array([[[1, 2, 3]],
                      [[4, 5, 6]]])  # shape: (2, 1, 3)

        B = np.ones((2, 4, 3))  # shape: (2, 4, 3)

        C = A + B  # A广播为 (2, 4, 3)

        self.assertEqual(C.shape, (2, 4, 3))

        # 验证第一行的所有列都相同（因为A的第一维是1）
        for j in range(4):
            np.testing.assert_array_equal(C[0, j, :], [2, 3, 4])
            np.testing.assert_array_equal(C[1, j, :], [5, 6, 7])

    def test_broadcast_sum_axis(self):
        """测试广播与求和轴的对应关系"""
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])  # (2, 3)
        B = np.array([10, 20, 30])  # (3,)

        # 验证广播逻辑：B在轴0上被复制
        B_broadcasted = np.broadcast_to(B, A.shape)

        self.assertEqual(B_broadcasted.shape, (2, 3))
        np.testing.assert_array_equal(B_broadcasted[0, :], [10, 20, 30])
        np.testing.assert_array_equal(B_broadcasted[1, :], [10, 20, 30])

    def test_broadcast_arithmetic_operations(self):
        """测试各种算术运算的广播"""
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        B = np.array([2, 3, 4])

        # 加法
        add_result = A + B
        expected_add = np.array([[3, 5, 7],
                                 [6, 8, 10]])
        np.testing.assert_array_equal(add_result, expected_add)

        # 乘法
        mul_result = A * B
        expected_mul = np.array([[2, 6, 12],
                                 [8, 15, 24]])
        np.testing.assert_array_equal(mul_result, expected_mul)

        # 除法
        div_result = A / B
        expected_div = np.array([[0.5, 2 / 3, 0.75],
                                 [2.0, 5 / 3, 1.5]])
        np.testing.assert_array_almost_equal(div_result, expected_div)

    def test_broadcast_transpose_corrected(self):
        """测试转置后的广播（修正版）"""
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])  # (2, 3)
        B = np.array([[10], [20]])  # (2, 1)

        # 转置后
        A_T = A.T  # (3, 2)
        B_T = B.T  # (1, 2)

        # 实际广播计算：B_T 被复制到3行
        # B_T broadcast_to (3, 2) = [[10, 20],
        #                            [10, 20],
        #                            [10, 20]]
        C = A_T + B_T  # 广播为 (3, 2)

        # 正确的结果应该是：
        expected = np.array([[1 + 10, 4 + 20],  # 第一行
                             [2 + 10, 5 + 20],  # 第二行
                             [3 + 10, 6 + 20]])  # 第三行

        np.testing.assert_array_equal(C, expected)
        print(f"A_T = \n{A_T}")
        print(f"B_T = \n{B_T}")
        print(f"C = A_T + B_T = \n{C}")

    def test_broadcast_assign(self):
        """测试广播赋值"""
        A = np.zeros((2, 3))
        B = np.array([1, 2, 3])

        # 使用广播进行赋值
        A[:] = B  # B广播到A的形状

        expected = np.array([[1, 2, 3],
                             [1, 2, 3]])
        np.testing.assert_array_equal(A, expected)


class TestNumpyBroadcastFunctions(unittest.TestCase):
    """测试NumPy的广播相关函数"""

    def test_broadcast_to(self):
        """测试np.broadcast_to函数"""
        A = np.array([1, 2, 3])  # (3,)

        # 广播到 (2, 3)
        B = np.broadcast_to(A, (2, 3))

        self.assertEqual(B.shape, (2, 3))
        np.testing.assert_array_equal(B[0, :], [1, 2, 3])
        np.testing.assert_array_equal(B[1, :], [1, 2, 3])

        # 测试无效广播
        with self.assertRaises(ValueError):
            np.broadcast_to(A, (2, 4))

    def test_broadcast_arrays(self):
        """测试np.broadcast_arrays函数"""
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])  # (2, 3)
        B = np.array([10, 20, 30])  # (3,)

        # 获取广播后的数组
        A_broadcast, B_broadcast = np.broadcast_arrays(A, B)

        self.assertEqual(A_broadcast.shape, (2, 3))
        self.assertEqual(B_broadcast.shape, (2, 3))

        # 验证B被正确广播
        np.testing.assert_array_equal(B_broadcast[0, :], [10, 20, 30])
        np.testing.assert_array_equal(B_broadcast[1, :], [10, 20, 30])


def run_broadcast_example():
    """运行原始示例并打印结果"""
    print("=" * 50)
    print("原始示例演示：")
    print("=" * 50)

    A = np.array([[1, 2, 3],  # shape: (2, 3)
                  [4, 5, 6]])

    B = np.array([10, 20, 30])  # shape: (3,) → 广播为 (2, 3)

    print(f"A = \n{A}")
    print(f"A.shape = {A.shape}")
    print(f"\nB = {B}")
    print(f"B.shape = {B.shape}")

    C = A + B  # 每个元素: C[i,j] = A[i,j] + B[j]

    print(f"\nA + B = \n{C}")
    print(f"(A + B).shape = {C.shape}")

    # 验证计算过程
    print("\n验证计算过程：")
    for i in range(2):
        for j in range(3):
            print(f"C[{i},{j}] = A[{i},{j}] + B[{j}] = {A[i, j]} + {B[j]} = {C[i, j]}")


if __name__ == '__main__':
    # 运行示例
    run_broadcast_example()

    print("\n" + "=" * 50)
    print("运行单元测试：")
    print("=" * 50)

    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTensorBroadcasting)
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestNumpyBroadcastFunctions))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
