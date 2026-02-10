import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Typography,
  Box,
} from '@mui/material';
import { format } from 'date-fns';

const ChangePointsTable = ({ data, events, onRefresh }) => {
  if (!data || data.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No change points data available</Typography>
      </Paper>
    );
  }

  const getImpactColor = (magnitude) => {
    switch (magnitude?.toLowerCase()) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const getDirectionColor = (difference) => {
    return difference > 0 ? 'success' : 'error';
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          Detected Change Points
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {data.length} change points detected
        </Typography>
      </Box>

      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Date</TableCell>
              <TableCell>Mean Before</TableCell>
              <TableCell>Mean After</TableCell>
              <TableCell>Difference</TableCell>
              <TableCell>Impact</TableCell>
              <TableCell>Probability</TableCell>
              <TableCell>Effect Size</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(data || []).map((cp) => (
              <TableRow key={cp?.change_point_id || cp?.id || Math.random()}>
                <TableCell>{cp?.change_point_id || cp?.id || 'N/A'}</TableCell>
                <TableCell>
                  {cp?.change_date ? format(new Date(cp.change_date), 'MMM dd, yyyy') : 'N/A'}
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={`${cp?.mean_before?.toFixed(3) || '0.000'}%`}
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={`${cp?.mean_after?.toFixed(3) || '0.000'}%`}
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={`${cp?.mean_difference?.toFixed(3) || '0.000'}%`}
                    color={getDirectionColor(cp?.mean_difference)}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={cp?.impact_magnitude || 'Unknown'}
                    color={getImpactColor(cp?.impact_magnitude)}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={`${((cp?.probability_mu2_gt_mu1 || 0) * 100).toFixed(1)}%`}
                    color={cp?.probability_mu2_gt_mu1 > 0.8 ? 'success' : 'default'}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={cp?.effect_size?.toFixed(2) || '0.00'}
                    color={Math.abs(cp?.effect_size || 0) > 0.8 ? 'primary' : 'default'}
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default ChangePointsTable;
