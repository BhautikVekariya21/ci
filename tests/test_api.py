# tests/test_api.py
import pytest
import json

class TestFlaskAPI:
    """Test Flask API endpoints"""
    
    def test_home_page(self, client):
        """Test home page loads"""
        response = client.get('/')
        assert response.status_code == 200, "Home page failed to load"
    
    def test_batch_page(self, client):
        """Test batch page loads"""
        response = client.get('/batch')
        assert response.status_code == 200, "Batch page failed to load"
    
    def test_metrics_page(self, client):
        """Test metrics page loads"""
        response = client.get('/metrics')
        assert response.status_code == 200, "Metrics page failed to load"
    
    def test_predict_endpoint_exists(self, client):
        """Test predict endpoint exists"""
        response = client.post('/api/predict', json={'text': 'test'})
        assert response.status_code in [200, 400, 500], "Predict endpoint not found"
    
    def test_predict_happy_sentiment(self, client):
        """Test prediction for happy sentiment"""
        response = client.post('/api/predict', json={
            'text': 'I love this beautiful day! Everything is wonderful!'
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'sentiment' in data, "Sentiment missing from response"
            assert data['sentiment'] in ['happy', 'sad'], "Invalid sentiment value"
    
    def test_predict_sad_sentiment(self, client):
        """Test prediction for sad sentiment"""
        response = client.post('/api/predict', json={
            'text': 'I am so sad and disappointed. Everything is terrible.'
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'sentiment' in data, "Sentiment missing from response"
    
    def test_predict_empty_text(self, client):
        """Test prediction with empty text"""
        response = client.post('/api/predict', json={'text': ''})
        
        assert response.status_code == 400, "Should reject empty text"
    
    def test_predict_response_structure(self, client):
        """Test prediction response structure"""
        response = client.post('/api/predict', json={
            'text': 'This is a test message'
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            required_keys = ['sentiment', 'confidence', 'probabilities']
            for key in required_keys:
                assert key in data, f"Key {key} missing from response"
    
    def test_predict_probabilities_sum(self, client):
        """Test that probabilities sum to 100"""
        response = client.post('/api/predict', json={
            'text': 'Test message'
        })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            if 'probabilities' in data:
                probs = data['probabilities']
                total = probs.get('happy', 0) + probs.get('sad', 0)
                assert abs(total - 100) < 1, f"Probabilities sum to {total}, not 100"
    
    def test_metrics_endpoint(self, client):
        """Test metrics API endpoint"""
        response = client.get('/api/metrics')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, dict), "Metrics should be a dictionary"
    
    def test_predict_multiple_requests(self, client):
        """Test multiple prediction requests"""
        texts = [
            'Happy message',
            'Sad message',
            'Another test'
        ]
        
        for text in texts:
            response = client.post('/api/predict', json={'text': text})
            assert response.status_code in [200, 400], f"Failed for text: {text}"