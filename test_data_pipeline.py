"""
Comprehensive Data Pipeline Testing Suite
==========================================

This script provides unit tests for all data pipeline components.
"""

import unittest
import numpy as np
import cv2
import torch
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import IMAGE_SIZE
from src.preprocessing import load_image, normalize_image
from src.augmentations import get_train_transforms, get_val_transforms
from src.datasets import LungCancerDataset
from src.utils import set_seed, ensure_dir


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    @classmethod
    def setUpClass(cls):
        """Create test images."""
        ensure_dir('data/test')
        
        # Create test PNG image
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite('data/test/test_image.png', test_img)
    
    def test_load_image_png(self):
        """Test loading PNG image."""
        image = load_image('data/test/test_image.png', image_size=224)
        
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(0 <= image.min() <= 1)
        self.assertTrue(0 <= image.max() <= 1)
    
    def test_normalize_image(self):
        """Test image normalization."""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        normalized = normalize_image(image)
        
        self.assertEqual(normalized.shape, image.shape)
        # Check that values are centered around 0
        self.assertTrue(normalized.min() < 0)
        self.assertTrue(normalized.max() > 0)
    
    def test_image_size_flexibility(self):
        """Test different output sizes."""
        for size in [128, 224, 256]:
            image = load_image('data/test/test_image.png', image_size=size)
            self.assertEqual(image.shape, (size, size, 3))


class TestAugmentations(unittest.TestCase):
    """Test augmentation pipeline."""
    
    def setUp(self):
        """Set up test image."""
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_train_transforms(self):
        """Test training augmentations."""
        transform = get_train_transforms(IMAGE_SIZE)
        augmented = transform(image=self.test_image)
        
        self.assertIn('image', augmented)
        
        aug_image = augmented['image']
        self.assertIsInstance(aug_image, torch.Tensor)
        self.assertEqual(aug_image.shape, (3, IMAGE_SIZE, IMAGE_SIZE))
    
    def test_val_transforms(self):
        """Test validation transforms (should be deterministic)."""
        transform = get_val_transforms(IMAGE_SIZE)
        
        # Apply twice - should give same result
        aug1 = transform(image=self.test_image)['image']
        aug2 = transform(image=self.test_image)['image']
        
        self.assertTrue(torch.equal(aug1, aug2))
    
    def test_augmentation_randomness(self):
        """Test that training augmentations are random."""
        set_seed(42)
        transform = get_train_transforms(IMAGE_SIZE)
        
        # Apply multiple times
        results = []
        for _ in range(5):
            aug = transform(image=self.test_image)['image']
            results.append(aug)
        
        # At least some should be different
        different_count = 0
        for i in range(1, len(results)):
            if not torch.equal(results[0], results[i]):
                different_count += 1
        
        self.assertGreater(different_count, 0, "Augmentations should produce different results")


class TestDataset(unittest.TestCase):
    """Test PyTorch Dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Create test dataset."""
        ensure_dir('data/test/dataset/cancer')
        ensure_dir('data/test/dataset/no_cancer')
        
        # Create test images
        for i in range(3):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(f'data/test/dataset/cancer/img_{i}.png', img)
            cv2.imwrite(f'data/test/dataset/no_cancer/img_{i}.png', img)
    
    def test_dataset_creation(self):
        """Test creating dataset."""
        image_paths = [
            'data/test/dataset/cancer/img_0.png',
            'data/test/dataset/no_cancer/img_0.png',
        ]
        labels = [1, 0]
        
        dataset = LungCancerDataset(
            image_paths=image_paths,
            labels=labels,
            transform=get_val_transforms(IMAGE_SIZE)
        )
        
        self.assertEqual(len(dataset), 2)
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        image_paths = ['data/test/dataset/cancer/img_0.png']
        labels = [1]
        
        dataset = LungCancerDataset(
            image_paths=image_paths,
            labels=labels,
            transform=get_val_transforms(IMAGE_SIZE)
        )
        
        image, label = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.item(), 1)
    
    def test_dataset_without_transform(self):
        """Test dataset without transforms."""
        image_paths = ['data/test/dataset/cancer/img_0.png']
        labels = [1]
        
        dataset = LungCancerDataset(
            image_paths=image_paths,
            labels=labels,
            transform=None
        )
        
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)


class TestDataLoaderIntegration(unittest.TestCase):
    """Test full pipeline integration."""
    
    def test_batch_processing(self):
        """Test processing a batch."""
        from torch.utils.data import DataLoader
        
        image_paths = [
            'data/test/dataset/cancer/img_0.png',
            'data/test/dataset/cancer/img_1.png',
            'data/test/dataset/no_cancer/img_0.png',
            'data/test/dataset/no_cancer/img_1.png',
        ]
        labels = [1, 1, 0, 0]
        
        dataset = LungCancerDataset(
            image_paths=image_paths,
            labels=labels,
            transform=get_val_transforms(IMAGE_SIZE)
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get first batch
        images, labels_batch = next(iter(dataloader))
        
        self.assertEqual(images.shape, (2, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(labels_batch.shape, (2,))
    
    def test_full_epoch(self):
        """Test iterating through full epoch."""
        from torch.utils.data import DataLoader
        
        image_paths = [f'data/test/dataset/cancer/img_{i}.png' for i in range(3)]
        labels = [1] * 3
        
        dataset = LungCancerDataset(
            image_paths=image_paths,
            labels=labels,
            transform=get_val_transforms(IMAGE_SIZE)
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        total_samples = 0
        for images, labels in dataloader:
            total_samples += images.size(0)
        
        self.assertEqual(total_samples, 3)


def run_tests():
    """Run all tests."""
    print("="*70)
    print("DATA PIPELINE UNIT TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestAugmentations))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoaderIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Data pipeline is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED. Please review the errors above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
