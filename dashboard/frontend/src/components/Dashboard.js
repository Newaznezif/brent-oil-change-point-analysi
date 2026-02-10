import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Button,
  Tabs,
  Tab,
} from '@mui/material';
import PriceChart from './PriceChart';
import ReturnsChart from './ReturnsChart';
import ChangePointsTable from './ChangePointsTable';
import EventsTable from './EventsTable';
import RegimesAnalysis from './RegimesAnalysis';
import StatisticsPanel from './StatisticsPanel';
import axios from 'axios';

const Dashboard = ({ initialData }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(initialData || {});
  const [priceData, setPriceData] = useState([]);
  const [returnsData, setReturnsData] = useState([]);
  const [changePoints, setChangePoints] = useState([]);
  const [events, setEvents] = useState([]);

  useEffect(() => {
    if (initialData) {
      setDashboardStats(initialData);
      loadAllData();
    }
  }, [initialData]);

  const loadAllData = async () => {
    setLoading(true);
    setError(null);

    try {
      // 1. Syncing with backend/api.py endpoints
      const [pricesRes, cpRes, eventsRes, returnsRes] = await Promise.all([
        axios.get('http://127.0.0.1:5000/api/prices'),
        axios.get('http://127.0.0.1:5000/api/change-points'),
        axios.get('http://127.0.0.1:5000/api/events'),
        axios.get('http://127.0.0.1:5000/api/returns')
      ]);

      // 2. Distributing data
      setPriceData(pricesRes.data);
      setChangePoints(cpRes.data);
      setEvents(eventsRes.data);
      setReturnsData(returnsRes.data);

      // 3. Update dashboard stats from backend summary if available, 
      // or calculate from components
      const summaryRes = await axios.get('http://127.0.0.1:5000/api/dashboard/summary');
      setDashboardStats(summaryRes.data);

    } catch (err) {
      setError(`Backend Connection Error: ${err.message}. Ensure Flask is running on port 5000.`);
      console.error('Error loading dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };


  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    loadAllData();
  };

  if (loading && !dashboardStats.key_metrics) {
    return (
      <Box sx={{ width: '100%', textAlign: 'center', mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Loading Dashboard...
        </Typography>
        <LinearProgress sx={{ mt: 2 }} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
        <Button onClick={handleRefresh} sx={{ ml: 2 }}>
          Retry
        </Button>
      </Alert>
    );
  }

  return (
    <Box>
      {/* Dashboard Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Brent Oil Analysis Dashboard
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Bayesian Change Point Detection & Market Regime Analysis
          </Typography>
        </Box>
        <Button
          variant="outlined"
          onClick={handleRefresh}
          disabled={loading}
        >
          {loading ? 'Refreshing...' : 'Refresh Data'}
        </Button>
      </Box>

      {/* Stats Summary */}
      {dashboardStats.key_metrics && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Total Days
                </Typography>
                <Typography variant="h5">
                  {dashboardStats.key_metrics.total_days?.toLocaleString() || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Change Points
                </Typography>
                <Typography variant="h5">
                  {dashboardStats.key_metrics.change_points_count || 0}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Avg Return
                </Typography>
                <Typography variant="h5" color={
                  dashboardStats.key_metrics.return_stats?.mean > 0 ? 'success.main' : 'error.main'
                }>
                  {dashboardStats.key_metrics.return_stats?.mean?.toFixed(3) || '0.000'}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Volatility
                </Typography>
                <Typography variant="h5">
                  {dashboardStats.key_metrics.return_stats?.std?.toFixed(3) || '0.000'}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Navigation Tabs */}
      <Paper sx={{ mb: 4 }}>
        <Tabs value={activeTab} onChange={handleTabChange} centered>
          <Tab label="Price Analysis" />
          <Tab label="Change Points" />
          <Tab label="Events" />
          <Tab label="Regimes" />
          <Tab label="Statistics" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Box>
        {activeTab === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <PriceChart data={priceData} changePoints={changePoints} />
            </Grid>
            <Grid item xs={12}>
              <ReturnsChart data={returnsData} changePoints={changePoints} />
            </Grid>
          </Grid>
        )}

        {activeTab === 1 && (
          <ChangePointsTable
            data={changePoints}
            events={events}
            onRefresh={handleRefresh}
          />
        )}

        {activeTab === 2 && (
          <EventsTable
            data={events}
            changePoints={changePoints}
            onRefresh={handleRefresh}
          />
        )}

        {activeTab === 3 && (
          <RegimesAnalysis
            onRefresh={handleRefresh}
          />
        )}

        {activeTab === 4 && (
          <StatisticsPanel
            stats={dashboardStats.key_metrics}
            onRefresh={handleRefresh}
          />
        )}
      </Box>

      {/* Last Updated */}
      {dashboardStats.last_updated && (
        <Box sx={{ mt: 4, pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="caption" color="text.secondary">
            Last updated: {dashboardStats.last_updated}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default Dashboard;
