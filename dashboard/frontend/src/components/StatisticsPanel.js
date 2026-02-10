import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Divider,
} from '@mui/material';

const StatisticsPanel = ({ stats, onRefresh }) => {
  if (!stats) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No statistics available</Typography>
      </Paper>
    );
  }

  const renderStatItem = (label, value, format = 'standard') => {
    let formattedValue = value;

    if (format === 'percentage' && typeof value === 'number') {
      formattedValue = `${value.toFixed(3)}%`;
    } else if (format === 'currency' && typeof value === 'number') {
      formattedValue = `$${value.toFixed(2)}`;
    } else if (typeof value === 'number') {
      formattedValue = value.toFixed(3);
    }

    return (
      <TableRow>
        <TableCell component="th" scope="row">
          <Typography variant="body2">{label}</Typography>
        </TableCell>
        <TableCell align="right">
          <Typography variant="body2" fontWeight="medium">
            {formattedValue}
          </Typography>
        </TableCell>
      </TableRow>
    );
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Statistical Summary
      </Typography>

      <Grid container spacing={3}>
        {/* Date Range */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Date Range
              </Typography>
              <Typography variant="h6">
                {stats.date_range?.start || 'N/A'} to {stats.date_range?.end || 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Total Days: {stats.total_days?.toLocaleString() || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Price Statistics */}
        {stats?.price_stats && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Price Statistics
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      {renderStatItem('Mean', stats?.price_stats?.mean, 'currency')}
                      {renderStatItem('Median', stats?.price_stats?.median, 'currency')}
                      {renderStatItem('Minimum', stats?.price_stats?.min, 'currency')}
                      {renderStatItem('Maximum', stats?.price_stats?.max, 'currency')}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Return Statistics */}
        {stats?.return_stats && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Return Statistics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <TableContainer>
                      <Table size="small">
                        <TableBody>
                          {renderStatItem('Mean Return', stats?.return_stats?.mean, 'percentage')}
                          {renderStatItem('Volatility', stats?.return_stats?.std, 'percentage')}
                          {renderStatItem('Minimum', stats?.return_stats?.min, 'percentage')}
                          {renderStatItem('Maximum', stats?.return_stats?.max, 'percentage')}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TableContainer>
                      <Table size="small">
                        <TableBody>
                          {renderStatItem('Skewness', stats?.return_stats?.skewness)}
                          {renderStatItem('Kurtosis', stats?.return_stats?.kurtosis)}
                          {renderStatItem('Change Points', stats?.change_points_count)}
                          {renderStatItem('Events', stats?.events_count)}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Analysis Summary */}
      <Box sx={{ mt: 4, pt: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="subtitle1" gutterBottom>
          Analysis Summary
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          This dashboard provides Bayesian change point detection analysis on Brent oil prices.
          The system identifies structural breaks in price behavior and associates them with
          geopolitical events to provide actionable insights for market participants.
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {stats.change_points_count || 0}
                </Typography>
                <Typography variant="caption">
                  Change Points
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {stats.events_count || 0}
                </Typography>
                <Typography variant="caption">
                  Events
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {stats.total_days?.toLocaleString() || 'N/A'}
                </Typography>
                <Typography variant="caption">
                  Trading Days
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6} md={3}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color={
                  stats.return_stats?.mean > 0 ? 'success.main' : 'error.main'
                }>
                  {stats.return_stats?.mean?.toFixed(3) || '0.000'}%
                </Typography>
                <Typography variant="caption">
                  Avg Daily Return
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default StatisticsPanel;
