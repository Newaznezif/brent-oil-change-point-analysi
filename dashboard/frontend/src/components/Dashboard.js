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

  // Date filter state
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // Highlighting state for interactions
  const [highlightedDate, setHighlightedDate] = useState(null);

  useEffect(() => {
    if (initialData) {
      setDashboardStats(initialData);
      loadAllData();
    }
  }, [initialData]);

  const loadAllData = async (start = startDate, end = endDate) => {
    setLoading(true);
    setError(null);

    try {
      const params = {};
      if (start) params.start_date = start;
      if (end) params.end_date = end;

      // 1. Syncing with backend/api.py endpoints with potential filters
      const [pricesRes, cpRes, eventsRes, returnsRes] = await Promise.all([
        axios.get('http://127.0.0.1:5000/api/prices', { params }),
        axios.get('http://127.0.0.1:5000/api/change-points'),
        axios.get('http://127.0.0.1:5000/api/events'),
        axios.get('http://127.0.0.1:5000/api/returns')
      ]);

      // 2. Distributing data
      setPriceData(pricesRes.data);
      setChangePoints(cpRes.data);
      setEvents(eventsRes.data);
      setReturnsData(returnsRes.data);

      // 3. Update dashboard stats
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

  const handleFilterApply = () => {
    loadAllData(startDate, endDate);
  };

  const handleDateHighlight = (date) => {
    setHighlightedDate(date);
    setActiveTab(0); // Switch to Price Analysis when an item is clicked
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
      <Box sx={{ mb: 4, display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between', alignItems: { xs: 'flex-start', md: 'center' }, gap: 2 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Brent Oil Analysis Dashboard
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Bayesian Change Point Detection & Market Regime Analysis
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <Box component="span" sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Typography variant="body2">From:</Typography>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ padding: '5px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
            <Typography variant="body2">To:</Typography>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ padding: '5px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
            <Button variant="contained" size="small" onClick={handleFilterApply}>Apply</Button>
          </Box>
          <Button
            variant="outlined"
            onClick={handleRefresh}
            disabled={loading}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </Button>
        </Box>
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
        <Tabs value={activeTab} onChange={handleTabChange} centered variant="scrollable" scrollButtons="auto">
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
              <PriceChart
                data={priceData}
                changePoints={changePoints}
                highlightedDate={highlightedDate}
              />
            </Grid>
            <Grid item xs={12}>
              <ReturnsChart
                data={returnsData}
                changePoints={changePoints}
                highlightedDate={highlightedDate}
              />
            </Grid>
          </Grid>
        )}

        {activeTab === 1 && (
          <ChangePointsTable
            data={changePoints}
            events={events}
            onRefresh={handleRefresh}
            onSelectDate={handleDateHighlight}
          />
        )}

        {activeTab === 2 && (
          <EventsTable
            data={events}
            changePoints={changePoints}
            onRefresh={handleRefresh}
            onSelectDate={handleDateHighlight}
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
