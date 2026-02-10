import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Paper, Typography, Box } from '@mui/material';

const PriceChart = ({ data, changePoints, highlightedDate }) => {
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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Brent Oil Price History
        </Typography>
        {highlightedDate && (
          <Typography variant="body2" color="primary" sx={{ fontWeight: 'bold' }}>
            Highlighted: {highlightedDate}
          </Typography>
        )}
      </Box>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10 }}
            angle={-45}
            textAnchor="end"
            height={70}
            interval="preserveStartEnd"
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

          {/* Change Points as Reference Lines */}
          {changePoints && changePoints.map((cp, idx) => (
            <ReferenceLine
              key={`cp-${idx}`}
              x={cp.change_date}
              stroke="#ff9800"
              label={{ position: 'top', value: 'CP', fill: '#ff9800', fontSize: 10 }}
              strokeDasharray="3 3"
            />
          ))}

          {/* Highlighted Date Line */}
          {highlightedDate && (
            <ReferenceLine
              x={highlightedDate}
              stroke="#f44336"
              strokeWidth={3}
              label={{ position: 'top', value: 'SELECTED', fill: '#f44336', fontSize: 12, fontWeight: 'bold' }}
            />
          )}

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
