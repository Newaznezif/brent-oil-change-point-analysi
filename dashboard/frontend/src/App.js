import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import { Container, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import Header from './components/Header';

const theme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    background: { default: '#f5f5f5' },
  },
});

function App() {
  const [dataLoaded, setDataLoaded] = useState(false);
  const [dashboardData, setDashboardData] = useState(null);

  useEffect(() => {
    // Sync points to the specific status/summary endpoint
    fetch('http://127.0.0.1:5000/api/dashboard/summary')
      .then(response => {
        if (!response.ok) throw new Error('Backend not responding');
        return response.json();
      })
      .then(data => {
        setDashboardData(data);
        setDataLoaded(true);
      })
      .catch(error => {
        console.error('Error loading dashboard data:', error);
        // Fallback stats so the UI can still render its structure
        setDashboardData({
          key_metrics: { total_days: 0, change_points_count: 0 },
          last_updated: new Date().toLocaleString()
        });
        setDataLoaded(true);
      });
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="App">
        <Header />
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          {dataLoaded ? (
            <Dashboard initialData={dashboardData} />
          ) : (
            <div style={{ textAlign: 'center', padding: '40px' }}>
              Loading Brent Oil Analysis Dashboard...
            </div>
          )}
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default App;