#!/usr/bin/env python3
"""
Test suite for type hints and error handling improvements in WhisperX.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisperx.utils import (
    make_safe, exact_div, str2bool, optional_int, optional_float
)


class TestUtilsFunctions(unittest.TestCase):
    """Test improved utility functions with type hints and error handling."""
    
    def test_make_safe_string(self):
        """Test make_safe function with various inputs."""
        # Test regular ASCII string
        result = make_safe("Hello World")
        self.assertEqual(result, "Hello World")
        
        # Test Unicode string
        result = make_safe("Hello 世界")
        self.assertIsInstance(result, str)
    
    def test_exact_div_valid(self):
        """Test exact_div with valid inputs."""
        self.assertEqual(exact_div(10, 2), 5)
        self.assertEqual(exact_div(100, 10), 10)
        self.assertEqual(exact_div(0, 5), 0)
    
    def test_exact_div_invalid(self):
        """Test exact_div with invalid inputs."""
        # Test non-exact division
        with self.assertRaises(AssertionError):
            exact_div(10, 3)
        
        # Test division by zero
        with self.assertRaises(ZeroDivisionError):
            exact_div(10, 0)
    
    def test_str2bool_valid(self):
        """Test str2bool with valid inputs."""
        self.assertTrue(str2bool("True"))
        self.assertFalse(str2bool("False"))
    
    def test_str2bool_invalid(self):
        """Test str2bool with invalid inputs."""
        with self.assertRaises(ValueError) as context:
            str2bool("true")  # lowercase
        self.assertIn("Expected one of", str(context.exception))
        
        with self.assertRaises(ValueError):
            str2bool("1")
    
    def test_optional_int_valid(self):
        """Test optional_int with valid inputs."""
        self.assertEqual(optional_int("42"), 42)
        self.assertEqual(optional_int("-10"), -10)
        self.assertIsNone(optional_int("None"))
    
    def test_optional_int_invalid(self):
        """Test optional_int with invalid inputs."""
        with self.assertRaises(ValueError) as context:
            optional_int("not_a_number")
        self.assertIn("Cannot convert", str(context.exception))
        
        with self.assertRaises(ValueError):
            optional_int("3.14")  # Float string
    
    def test_optional_float_valid(self):
        """Test optional_float with valid inputs."""
        self.assertEqual(optional_float("3.14"), 3.14)
        self.assertEqual(optional_float("-2.5"), -2.5)
        self.assertIsNone(optional_float("None"))
        self.assertEqual(optional_float("42"), 42.0)  # Integer string
    
    def test_optional_float_invalid(self):
        """Test optional_float with invalid inputs."""
        with self.assertRaises(ValueError) as context:
            optional_float("not_a_float")
        self.assertIn("Cannot convert", str(context.exception))


class TestFileHandling(unittest.TestCase):
    """Test improved file handling in pyannote VAD."""
    
    def test_model_file_context_manager(self):
        """Verify that file handles are properly closed using context managers."""
        # This test verifies the code structure rather than execution
        # since we don't have the actual model files
        import inspect
        from whisperx.vads.pyannote import load_vad_model
        
        # Get the source code of the function
        source = inspect.getsource(load_vad_model)
        
        # Check that we're using 'with open' instead of bare 'open'
        self.assertIn("with open", source)
        self.assertNotIn("open(model_fp, \"rb\").read()", source)


class TestErrorHandling(unittest.TestCase):
    """Test improved error handling in alignment module."""
    
    def test_specific_exception_handling(self):
        """Verify that specific exceptions are caught instead of bare Exception."""
        import inspect
        from whisperx.alignment import load_align_model
        
        # Get the source code
        source = inspect.getsource(load_align_model)
        
        # Check that we're catching specific exceptions
        self.assertIn("except (OSError, RuntimeError, KeyError)", source)
        # Verify we're not using bare except Exception
        self.assertNotIn("except Exception as e:", source)
        # Check that we're using proper logging
        self.assertIn("logger", source)


class TestTypeHints(unittest.TestCase):
    """Test that type hints are properly added."""
    
    def test_function_annotations(self):
        """Verify that functions have proper type annotations."""
        from whisperx.utils import exact_div, str2bool, optional_int, optional_float
        
        # Check that functions have annotations
        self.assertTrue(hasattr(exact_div, '__annotations__'))
        self.assertTrue(hasattr(str2bool, '__annotations__'))
        self.assertTrue(hasattr(optional_int, '__annotations__'))
        self.assertTrue(hasattr(optional_float, '__annotations__'))
        
        # Verify return type annotations
        self.assertIn('return', exact_div.__annotations__)
        self.assertIn('return', str2bool.__annotations__)
        self.assertIn('return', optional_int.__annotations__)
        self.assertIn('return', optional_float.__annotations__)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)