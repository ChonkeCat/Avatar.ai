import numpy as np
import sys
import os

# Add the necessary paths to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cnn_utils import LeakyRelU
from model.Conv2D import Conv2D  # Note: You might need to rename this file to avoid import issues

def test_conv2d_initialization():
    """Test Conv2D layer initialization"""
    print("Testing Conv2D initialization...")
    
    # Test with valid padding
    conv_layer = Conv2D(
        filter_size=(3, 3),
        n_filters=32,
        activation_func=LeakyRelU,
        input_shape=(1, 28, 28, 1),  # batch_size=1, 28x28, 1 channel
        first=True,
        padding="valid",
        stride=1
    )
    
    output_shape = conv_layer.initialize((1, 28, 28, 1))
    print(f"Input shape: (1, 28, 28, 1)")
    print(f"Output shape with valid padding: {output_shape}")
    print(f"Weight shape: {conv_layer.W.shape}")
    print(f"Bias shape: {conv_layer.b.shape}")
    
    # Test with same padding
    conv_layer_same = Conv2D(
        filter_size=(3, 3),
        n_filters=32,
        activation_func=LeakyRelU,
        input_shape=(1, 28, 28, 1),
        first=True,
        padding="same",
        stride=1
    )
    
    output_shape_same = conv_layer_same.initialize((1, 28, 28, 1))
    print(f"Output shape with same padding: {output_shape_same}")
    
    print("✓ Initialization test passed!\n")

def test_conv2d_forward_pass():
    """Test Conv2D forward pass with simple input"""
    print("Testing Conv2D forward pass...")
    
    # Create a simple test input (2x2 image with 1 channel)
    test_input = np.array([[
        [[1], [2]],
        [[3], [4]]
    ]])  # shape: (1, 2, 2, 1)
    
    # Create Conv2D layer with 1 filter of size 2x2
    conv_layer = Conv2D(
        filter_size=(2, 2),
        n_filters=1,
        activation_func=LeakyRelU,
        input_shape=(1, 2, 2, 1),
        first=True,
        padding="valid",
        stride=1
    )
    
    # Manually set weights and bias for predictable output
    conv_layer.initialize((1, 2, 2, 1))
    conv_layer.W = np.ones((2, 2, 1, 1))  # All weights = 1
    conv_layer.b = np.zeros(1)  # Bias = 0
    
    # Perform forward pass
    output = conv_layer.forward(test_input)
    
    print(f"Input:\n{test_input.reshape(2, 2)}")
    print(f"Weights (all ones):\n{conv_layer.W.reshape(2, 2)}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output.flatten()}")
    
    # Expected output: 1*1 + 2*1 + 3*1 + 4*1 = 10
    expected_output = np.array([10.0])
    actual_output = output.flatten()
    
    assert np.allclose(actual_output, expected_output), f"Expected {expected_output}, got {actual_output}"
    print("✓ Forward pass test passed!\n")

def test_conv2d_backward_pass():
    """Test Conv2D backward pass"""
    print("Testing Conv2D backward pass...")
    
    # Create test input
    test_input = np.random.randn(2, 4, 4, 3)  # batch_size=2, 4x4, 3 channels
    
    conv_layer = Conv2D(
        filter_size=(3, 3),
        n_filters=8,
        activation_func=LeakyRelU,
        input_shape=(2, 4, 4, 3),
        first=True,
        padding="valid",
        stride=1
    )
    
    conv_layer.initialize((2, 4, 4, 3))
    
    # Forward pass
    output = conv_layer.forward(test_input)
    print(f"Forward pass completed. Output shape: {output.shape}")
    
    # Create dummy gradient (same shape as output)
    dummy_gradient = np.ones_like(output)
    
    # Backward pass
    input_gradient = conv_layer.backward(dummy_gradient)
    
    print(f"Input gradient shape: {input_gradient.shape}")
    print(f"Weight gradient shape: {conv_layer.dW.shape}")
    print(f"Bias gradient shape: {conv_layer.db.shape}")
    
    # Check shapes
    assert input_gradient.shape == test_input.shape, "Input gradient shape mismatch"
    assert conv_layer.dW.shape == conv_layer.W.shape, "Weight gradient shape mismatch"
    assert conv_layer.db.shape == conv_layer.b.shape, "Bias gradient shape mismatch"
    
    print("✓ Backward pass test passed!\n")

def test_conv2d_gradient_flow():
    """Test that gradients flow properly through the layer"""
    print("Testing gradient flow...")
    
    # Create a simple network: Conv2D -> Flatten -> Dense equivalent
    test_input = np.random.randn(1, 5, 5, 1).astype(np.float32)
    
    conv_layer = Conv2D(
        filter_size=(3, 3),
        n_filters=2,
        activation_func=LeakyRelU,
        input_shape=(1, 5, 5, 1),
        first=True,
        padding="valid",
        stride=1
    )
    
    conv_layer.initialize((1, 5, 5, 1))
    
    # Forward pass
    conv_output = conv_layer.forward(test_input)
    print(f"Conv output shape: {conv_output.shape}")
    
    # Simulate a simple loss (mean squared error)
    target = np.ones_like(conv_output)
    loss = np.mean((conv_output - target) ** 2)
    print(f"Initial loss: {loss}")
    
    # Backward pass (gradient of MSE loss)
    dLdA = 2 * (conv_output - target) / conv_output.size
    input_gradient = conv_layer.backward(dLdA)
    
    # Check that gradients are not all zeros
    assert not np.allclose(conv_layer.dW, 0), "Weight gradients are all zero!"
    assert not np.allclose(conv_layer.db, 0), "Bias gradients are all zero!"
    assert not np.allclose(input_gradient, 0), "Input gradients are all zero!"
    
    print("Weight gradients range:", np.min(conv_layer.dW), "to", np.max(conv_layer.dW))
    print("Bias gradients range:", np.min(conv_layer.db), "to", np.max(conv_layer.db))
    print("✓ Gradient flow test passed!\n")

def test_conv2d_different_strides():
    """Test Conv2D with different stride values"""
    print("Testing different stride values...")
    
    test_input = np.random.randn(1, 6, 6, 1)
    
    for stride in [1, 2, 3]:
        conv_layer = Conv2D(
            filter_size=(3, 3),
            n_filters=4,
            activation_func=LeakyRelU,
            input_shape=(1, 6, 6, 1),
            first=True,
            padding="valid",
            stride=stride
        )
        
        output_shape = conv_layer.initialize((1, 6, 6, 1))
        output = conv_layer.forward(test_input)
        
        print(f"Stride {stride}: Input (1, 6, 6, 1) -> Output {output.shape}")
        
        # Verify output dimensions
        expected_height = (6 - 3) // stride + 1
        expected_width = (6 - 3) // stride + 1
        assert output.shape[1] == expected_height, f"Height mismatch for stride {stride}"
        assert output.shape[2] == expected_width, f"Width mismatch for stride {stride}"
    
    print("✓ Different strides test passed!\n")

def test_conv2d_multiple_channels():
    """Test Conv2D with multiple input and output channels"""
    print("Testing multiple channels...")
    
    # Input with 3 channels (like RGB image)
    test_input = np.random.randn(2, 8, 8, 3)  # batch_size=2, 8x8, 3 channels
    
    conv_layer = Conv2D(
        filter_size=(3, 3),
        n_filters=16,  # 16 output channels
        activation_func=LeakyRelU,
        input_shape=(2, 8, 8, 3),
        first=True,
        padding="valid",
        stride=1
    )
    
    output_shape = conv_layer.initialize((2, 8, 8, 3))
    output = conv_layer.forward(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight shape: {conv_layer.W.shape}")  # Should be (3, 3, 3, 16)
    
    # Verify shapes
    assert conv_layer.W.shape == (3, 3, 3, 16), "Weight shape incorrect for multiple channels"
    assert output.shape == (2, 6, 6, 16), "Output shape incorrect"
    
    # Test backward pass
    dummy_gradient = np.ones_like(output)
    input_gradient = conv_layer.backward(dummy_gradient)
    
    assert input_gradient.shape == test_input.shape, "Input gradient shape incorrect"
    
    print("✓ Multiple channels test passed!\n")

def test_conv2d_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    # Test with minimum valid input size
    test_input = np.random.randn(1, 3, 3, 1)
    
    conv_layer = Conv2D(
        filter_size=(3, 3),
        n_filters=1,
        activation_func=LeakyRelU,
        input_shape=(1, 3, 3, 1),
        first=True,
        padding="valid",
        stride=1
    )
    
    output_shape = conv_layer.initialize((1, 3, 3, 1))
    output = conv_layer.forward(test_input)
    
    print(f"Minimum input (3x3) with 3x3 filter -> Output shape: {output.shape}")
    assert output.shape == (1, 1, 1, 1), "Minimum size test failed"
    
    # Test with 'same' padding on odd-sized input
    test_input_odd = np.random.randn(1, 5, 5, 1)
    
    conv_layer_same = Conv2D(
        filter_size=(3, 3),
        n_filters=1,
        activation_func=LeakyRelU,
        input_shape=(1, 5, 5, 1),
        first=True,
        padding="same",
        stride=1
    )
    
    output_shape_same = conv_layer_same.initialize((1, 5, 5, 1))
    output_same = conv_layer_same.forward(test_input_odd)
    
    print(f"Input (5x5) with same padding -> Output shape: {output_same.shape}")
    assert output_same.shape == (1, 5, 5, 1), "Same padding test failed"
    
    print("✓ Edge cases test passed!\n")

def run_all_tests():
    """Run all Conv2D tests"""
    print("=" * 60)
    print("Starting Conv2D Layer Tests")
    print("=" * 60)
    
    try:
        test_conv2d_initialization()
        test_conv2d_forward_pass()
        test_conv2d_backward_pass()
        test_conv2d_gradient_flow()
        test_conv2d_different_strides()
        test_conv2d_multiple_channels()
        test_conv2d_edge_cases()
        
        print("=" * 60)
        print("🎉 ALL CONV2D TESTS PASSED! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()