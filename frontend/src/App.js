import {
    CheckCircle,
    DataUsage,
    Error,
    Info,
    Psychology,
    Security,
    TrendingUp,
    Warning
} from '@mui/icons-material';
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    CircularProgress,
    Container,
    Divider,
    Grid,
    LinearProgress,
    Paper,
    TextField,
    Typography
} from '@mui/material';
import axios from 'axios';
import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [healthStatus, setHealthStatus] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mouseData, setMouseData] = useState('');
  const [webLogData, setWebLogData] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/health');
      setHealthStatus(response.data);
    } catch (err) {
      setError('Backend server is not running. Please start the Flask server.');
    }
  };

  const loadSampleData = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5000/api/sample-data');
      setMouseData(JSON.stringify(response.data.mouse_movements, null, 2));
      setWebLogData(JSON.stringify(response.data.web_logs, null, 2));
      setError('');
    } catch (err) {
      setError('Failed to load sample data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const detectBot = async () => {
    if (!mouseData || !webLogData) {
      setError('Please provide both mouse movement and web log data');
      return;
    }

    try {
      setLoading(true);
      setError('');
      
      const requestData = {
        mouse_movements: JSON.parse(mouseData),
        web_logs: JSON.parse(webLogData)
      };

      const response = await axios.post('http://localhost:5000/api/detect', requestData);
      setDetectionResult(response.data);
    } catch (err) {
      setError('Detection failed: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  const getResultColor = (isBot) => {
    return isBot ? '#f44336' : '#4caf50';
  };

  const getResultIcon = (isBot) => {
    return isBot ? <Error /> : <CheckCircle />;
  };

  return (
    <div className="App">
      <Container maxWidth="lg">
        {/* Header */}
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h2" component="h1" gutterBottom>
            <Security sx={{ mr: 2, color: 'primary.main' }} />
            Bot Detection System
          </Typography>
          <Typography variant="h5" color="text.secondary" gutterBottom>
            Federated Learning for Advanced Bot Detection
          </Typography>
        </Box>

        {/* Health Status */}
        {healthStatus && (
          <Alert 
            severity={healthStatus.model_loaded ? 'success' : 'warning'} 
            sx={{ mb: 3 }}
          >
            {healthStatus.model_loaded 
              ? `Model loaded successfully with ${healthStatus.feature_count} features`
              : 'Model not loaded. Please check the backend.'
            }
          </Alert>
        )}

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        {/* Problem Section */}
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h4" gutterBottom>
              <Warning sx={{ mr: 1, color: 'warning.main' }} />
              The Problem: Grinch Bots
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>What are Grinch Bots?</Typography>
                <Typography paragraph>
                  Grinch bots are sophisticated automated programs that flood online stores during 
                  high-demand periods, automatically purchasing limited items before humans can access them.
                </Typography>
                <Box component="ul">
                  <Typography component="li">Use advanced techniques to mimic human behavior</Typography>
                  <Typography component="li">Resell items at inflated prices</Typography>
                  <Typography component="li">Cause frustration for legitimate customers</Typography>
                  <Typography component="li">Evolve rapidly to bypass detection methods</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>Why Traditional Detection Fails</Typography>
                <Box component="ul">
                  <Typography component="li">Bots evolve rapidly to bypass simple rules</Typography>
                  <Typography component="li">Individual retailers lack sufficient data</Typography>
                  <Typography component="li">Privacy concerns prevent data sharing</Typography>
                  <Typography component="li">Detection methods become outdated quickly</Typography>
                  <Typography component="li">False positives hurt legitimate customers</Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Solution Section */}
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h4" gutterBottom>
              <Psychology sx={{ mr: 1, color: 'primary.main' }} />
              The Solution: Federated Learning
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>How Federated Learning Works</Typography>
                <Box component="ol">
                  <Typography component="li">Each retailer trains on their own data</Typography>
                  <Typography component="li">Only model updates are shared (not raw data)</Typography>
                  <Typography component="li">Server combines updates from all retailers</Typography>
                  <Typography component="li">Improved model is distributed back to all retailers</Typography>
                  <Typography component="li">Raw customer data never leaves the organization</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>Benefits</Typography>
                <Box component="ul">
                  <Typography component="li">Privacy: Customer data stays within each organization</Typography>
                  <Typography component="li">Collaboration: Multiple retailers improve detection together</Typography>
                  <Typography component="li">Scalability: More participants = better detection</Typography>
                  <Typography component="li">Adaptability: Model continuously improves with new data</Typography>
                  <Typography component="li">Compliance: Meets data protection regulations</Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Detection Section */}
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h4" gutterBottom>
              <DataUsage sx={{ mr: 1, color: 'secondary.main' }} />
              Live Bot Detection
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>Test the Model</Typography>
                <Typography paragraph>
                  Upload mouse movement and web log data to test our bot detection system:
                </Typography>
                
                <TextField
                  fullWidth
                  multiline
                  rows={6}
                  label="Mouse Movement Data (JSON)"
                  value={mouseData}
                  onChange={(e) => setMouseData(e.target.value)}
                  sx={{ mb: 2 }}
                />
                
                <TextField
                  fullWidth
                  multiline
                  rows={6}
                  label="Web Log Data (JSON)"
                  value={webLogData}
                  onChange={(e) => setWebLogData(e.target.value)}
                  sx={{ mb: 2 }}
                />
                
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button 
                    variant="outlined" 
                    onClick={loadSampleData}
                    disabled={loading}
                    startIcon={<Info />}
                  >
                    Load Sample Data
                  </Button>
                  <Button 
                    variant="contained" 
                    onClick={detectBot}
                    disabled={loading}
                    startIcon={<TrendingUp />}
                  >
                    Detect Bot
                  </Button>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                {loading && (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <CircularProgress size={60} />
                    <Typography sx={{ mt: 2 }}>Analyzing behavior patterns...</Typography>
                  </Box>
                )}
                
                {detectionResult && (
                  <Paper 
                    elevation={3} 
                    sx={{ 
                      p: 3, 
                      backgroundColor: getResultColor(detectionResult.prediction === 1),
                      color: 'white'
                    }}
                  >
                    <Box sx={{ textAlign: 'center', mb: 3 }}>
                      {getResultIcon(detectionResult.prediction === 1)}
                      <Typography variant="h4" sx={{ mt: 1 }}>
                        {detectionResult.prediction_label}
                      </Typography>
                      <Typography variant="h6">
                        Confidence: {(detectionResult.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    
                    <Divider sx={{ backgroundColor: 'rgba(255,255,255,0.3)', my: 2 }} />
                    
                    <Typography variant="h6" gutterBottom>Probabilities</Typography>
                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography>Human</Typography>
                        <Typography>{(detectionResult.human_probability * 100).toFixed(1)}%</Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={detectionResult.human_probability * 100}
                        sx={{ height: 10, borderRadius: 5 }}
                      />
                    </Box>
                    
                    <Box sx={{ mb: 3 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography>Bot</Typography>
                        <Typography>{(detectionResult.bot_probability * 100).toFixed(1)}%</Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={detectionResult.bot_probability * 100}
                        sx={{ height: 10, borderRadius: 5 }}
                      />
                    </Box>
                    
                    <Typography variant="h6" gutterBottom>Key Features</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {detectionResult.top_features.map((feature, index) => (
                        <Chip
                          key={index}
                          label={`${feature.name}: ${(feature.importance * 100).toFixed(1)}%`}
                          size="small"
                          sx={{ backgroundColor: 'rgba(255,255,255,0.2)' }}
                        />
                      ))}
                    </Box>
                    
                    <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', mt: 2 }}>
                      Analysis completed at {new Date(detectionResult.timestamp).toLocaleString()}
                    </Typography>
                  </Paper>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Container>
    </div>
  );
}

export default App; 