import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Paper, Typography } from '@mui/material';

const PriceChart = ({ data, changePoints }) => {
  if (!data || !data.dates || data.dates.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No price data available</Typography>
      </Paper>
    );
  }

  // Format data for chart
  const chartData = (data?.dates || []).map((date, index) => ({
    date,
    price: data?.prices?.[index] || 0,
  }));

  // Find min and max for y-axis
  const prices = data?.prices || [];
  const minPrice = prices.length > 0 ? Math.min(...prices) * 0.95 : 0;
  const maxPrice = prices.length > 0 ? Math.max(...prices) * 1.05 : 100;

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Brent Oil Price History
      </Typography>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis
            domain={[minPrice, maxPrice]}
            tickFormatter={(value) => `$${value.toFixed(2)}`}
          />
          <Tooltip
            formatter={(value) => [`$${value.toFixed(2)}`, 'Price']}
            labelFormatter={(label) => `Date: ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#1976d2"
            strokeWidth={2}
            dot={false}
            name="Brent Oil Price"
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export default PriceChart;
