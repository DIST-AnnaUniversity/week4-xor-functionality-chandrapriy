import numpy as np
import xor

def test_xor():
    X = np.array([[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1],])
    w1 = np.array([[-2,1,1/2],[1,-1,1/2],])
    w2 = np.array([[1,1,1],])
    layer1 = xor.calculate_layer1(X[0],w1)
    out1 = np.array([[-1,-1],])
    np.testing.assert_array_equal(layer1, out1)
    layer2 = xor.calculate_layer2(layer1,w2)
    out2 = np.array([[-1]])
    np.testing.assert_array_equal(layer2, out2)
    ##############################
    layer1 = xor.calculate_layer1(X[1],w1)
    out1 = np.array([[1,-1],])
    np.testing.assert_array_equal(layer1, out1)
    layer2 = xor.calculate_layer2(layer1,w2)
    out2 = np.array([[1]])
    np.testing.assert_array_equal(layer2, out2)
    ####################################
    layer1 = xor.calculate_layer1(X[2],w1)
    out1 = np.array([[-1,1],])
    np.testing.assert_array_equal(layer1, out1)
    layer2 = xor.calculate_layer2(layer1,w2)
    out2 = np.array([[1]])
    np.testing.assert_array_equal(layer2, out2)
    ############################
    layer1 = xor.calculate_layer1(X[3],w1)
    out1 = np.array([[-1,-1],])
    np.testing.assert_array_equal(layer1, out1)
    layer2 = xor.calculate_layer2(layer1,w2)
    out2 = np.array([[-1]])
    np.testing.assert_array_equal(layer2, out2)
