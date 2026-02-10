import React from 'react';
import { AppBar, Toolbar, Typography, Box, Chip } from '@mui/material';
import OilBarrelIcon from '@mui/icons-material/OilBarrel';

const Header = () => {
  return (
    <AppBar position="static" sx={{ mb: 4 }}>
      <Toolbar>
        <OilBarrelIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Brent Oil Analysis Dashboard
        </Typography>
        <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 2 }}>
          <Chip 
            label="Bayesian Analysis" 
            color="secondary" 
            size="small" 
            variant="outlined"
          />
          <Chip 
            label="Change Point Detection" 
            color="primary" 
            size="small" 
            variant="outlined"
          />
          <Chip 
            label="Real-time Monitoring" 
            color="success" 
            size="small" 
            variant="outlined"
          />
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
