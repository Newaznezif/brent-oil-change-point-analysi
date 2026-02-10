import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { Paper, Typography } from '@mui/material';

const ReturnsChart = ({ data, changePoints, highlightedDate }) => {
  // 1. Data Safety Check & Transformation
  const chartData = React.useMemo(() => {
    if (!data) return [];

    // Case 1: Backend object structure { dates: [], returns: [] }
    if (data.dates && Array.isArray(data.dates) && data.returns && Array.isArray(data.returns)) {
      return data.dates.map((date, index) => ({
        date: date || 'N/A',
        return: data.returns[index] || 0
      }));
    }

    // Case 2: Array of objects [{ Date/date, Return/return/Price }, ...]
    if (Array.isArray(data)) {
      return data.map((item, index) => {
        let returnVal = 0;
        if (item?.Return !== undefined) {
          returnVal = item.Return;
        } else if (item?.return !== undefined) {
          returnVal = item.return;
        }
        // Fallback calculation if Prices are available
        else if (index > 0 && data[index - 1]?.Price && item?.Price) {
          returnVal = ((item.Price - data[index - 1].Price) / data[index - 1].Price) * 100;
        }

        return {
          date: item?.Date || item?.date || 'N/A',
          return: returnVal
        };
      });
    }

    return [];
  }, [data]);

  if (chartData.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No returns data available</Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 3, mt: 3 }}>
      <Typography variant="h6" gutterBottom>Daily Returns (%) - Volatility Analysis</Typography>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" hide={chartData.length > 200} tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
          <YAxis tickFormatter={(val) => `${val?.toFixed(1) || '0'}%`} />
          <Tooltip formatter={(val) => [`${val?.toFixed(3) || '0.000'}%`, 'Return']} />
          <Legend />

          {/* Change Points as Reference Lines */}
          {changePoints && changePoints.map((cp, idx) => (
            <ReferenceLine
              key={`cp-ret-${idx}`}
              x={cp.change_date}
              stroke="#ff9800"
              strokeDasharray="3 3"
            />
          ))}

          {/* Highlighted Date Line */}
          {highlightedDate && (
            <ReferenceLine
              x={highlightedDate}
              stroke="#f44336"
              strokeWidth={2}
            />
          )}

          <Bar dataKey="return" name="Daily Return">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.return >= 0 ? '#4caf50' : '#f44336'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export default ReturnsChart;