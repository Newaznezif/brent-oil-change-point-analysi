import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Button,
  Chip,
} from '@mui/material';
import axios from 'axios';

const RegimesAnalysis = ({ onRefresh }) => {
  const [regimes, setRegimes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadRegimes();
  }, []);

  const loadRegimes = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get('http://127.0.0.1:5000/api/regimes');
      setRegimes(response.data || []);
    } catch (err) {
      setError(`Error loading regimes: ${err.message}`);
      console.error('Error loading regimes:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRegimeColor = (meanReturn) => {
    return meanReturn > 0 ? 'success.main' : 'error.main';
  };

  if (loading) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography gutterBottom>Loading regimes analysis...</Typography>
        <LinearProgress />
      </Paper>
    );
  }

  if (error) {
    return (
      <Alert
        severity="error"
        sx={{ mt: 2 }}
        action={
          <Button color="inherit" size="small" onClick={loadRegimes}>
            Retry
          </Button>
        }
      >
        {error}
      </Alert>
    );
  }

  if (!regimes || regimes.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No regimes data available</Typography>
        <Button onClick={loadRegimes} sx={{ mt: 2 }}>
          Load Regimes
        </Button>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          Market Regimes Analysis
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {regimes.length} regimes identified
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {regimes.map((regime) => (
          <Grid item xs={12} md={6} lg={4} key={regime?.regime_id || Math.random()}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Regime {regime?.regime_id || 'N/A'}
                  {regime?.is_change_point && (
                    <Chip
                      size="small"
                      label="Change Point"
                      color="primary"
                      sx={{ ml: 1 }}
                    />
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {regime?.start_date || 'N/A'} to {regime?.end_date || 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {regime?.duration_days || 0} days
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="caption" display="block">
                      Mean Return
                    </Typography>
                    <Typography
                      variant="h6"
                      color={getRegimeColor(regime?.mean_return || 0)}
                    >
                      {regime?.mean_return?.toFixed(3) || '0.000'}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" display="block">
                      Volatility
                    </Typography>
                    <Typography variant="h6">
                      {regime?.volatility?.toFixed(3) || '0.000'}%
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="caption" display="block">
                      Observations
                    </Typography>
                    <Typography variant="body2">
                      {regime?.observations || 0} days
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default RegimesAnalysis;
