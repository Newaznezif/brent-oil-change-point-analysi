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

const EventsTable = ({ data, changePoints, onRefresh, onSelectDate }) => {
  if (!data || data.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No events data available</Typography>
      </Paper>
    );
  }

  const getImpactColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          Geopolitical Events
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {data.length} events recorded (Click row to highlight)
        </Typography>
      </Box>

      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Event</TableCell>
              <TableCell>Date</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Impact</TableCell>
              <TableCell>Description</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(data || []).map((event, index) => (
              <TableRow
                key={event?.event_id || index}
                hover
                onClick={() => onSelectDate && onSelectDate(event.date)}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell>
                  <Typography fontWeight="medium">
                    {event?.event_name || 'Unnamed Event'}
                  </Typography>
                </TableCell>
                <TableCell>
                  {event?.date ? format(new Date(event.date), 'MMM dd, yyyy') : 'N/A'}
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={event?.event_type || 'Unknown'}
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    label={event?.impact_level || 'Unknown'}
                    color={getImpactColor(event?.impact_level)}
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ maxWidth: 300 }}>
                    {event?.description || 'No description available'}
                  </Typography>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default EventsTable;
