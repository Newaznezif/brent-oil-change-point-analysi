"""
API endpoints for Brent Oil Analysis Dashboard
"""
from flask import Blueprint, jsonify, request, current_app
import os
import json
import pandas as pd
from datetime import datetime, timedelta

api_bp = Blueprint('api', __name__)

def get_dm():
    """Helper to get data manager from current_app"""
    return current_app.config['DATA_MANAGER']

@api_bp.route('/prices')
def get_prices():
    """Get Brent oil prices with optional date range"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        prices = get_dm().get_prices(start_date, end_date)
        return jsonify(prices)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/returns')
def get_returns():
    """Get log returns data"""
    try:
        returns = get_dm().get_returns()
        return jsonify(returns)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/change-points')
def get_change_points():
    """Get detected change points"""
    try:
        change_points = get_dm().get_change_points()
        return jsonify(change_points)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/events')
def get_events():
    """Get geopolitical events"""
    try:
        events = get_dm().get_events()
        return jsonify(events)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/events/near-change-points')
def get_events_near_change_points():
    """Get events near detected change points"""
    try:
        max_days = request.args.get('max_days', default=30, type=int)
        events = get_dm().get_events_near_change_points(max_days)
        return jsonify(events)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/regimes')
def get_regimes():
    """Get market regimes based on change points"""
    try:
        regimes = get_dm().get_regimes()
        return jsonify(regimes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/statistics')
def get_statistics():
    """Get statistical summary"""
    try:
        stats = get_dm().get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analysis/single-change-point')
def get_single_change_point_analysis():
    """Get single change point analysis results"""
    try:
        analysis = get_dm().get_single_change_point_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analysis/multiple-change-points')
def get_multiple_change_points_analysis():
    """Get multiple change points analysis"""
    try:
        analysis = get_dm().get_multiple_change_points_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/impact-analysis')
def get_impact_analysis():
    """Get impact analysis of change points"""
    try:
        impact = get_dm().get_impact_analysis()
        return jsonify(impact)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/recommendations')
def get_recommendations():
    """Get recommendations based on analysis"""
    try:
        recommendations = get_dm().get_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/simulate', methods=['POST'])
def simulate_analysis():
    """Run simulation with different parameters"""
    try:
        data = request.get_json()
        
        # Get simulation parameters
        window_size = data.get('window_size', 60)
        threshold = data.get('threshold', 0.8)
        
        # Run simulation (simplified for now)
        simulation_results = {
            'window_size': window_size,
            'threshold': threshold,
            'simulated_change_points': [],
            'message': 'Simulation would run here with real implementation'
        }
        
        return jsonify(simulation_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/export/<data_type>')
def export_data(data_type):
    """Export data as CSV"""
    try:
        valid_types = ['prices', 'returns', 'change_points', 'events']
        if data_type not in valid_types:
            return jsonify({'error': f'Invalid data type. Must be one of: {valid_types}'}), 400
        
        # Get data based on type
        if data_type == 'prices':
            data = get_dm().get_prices()
            filename = 'brent_prices_export.csv'
        elif data_type == 'returns':
            data = get_dm().get_returns()
            filename = 'brent_returns_export.csv'
        elif data_type == 'change_points':
            data = get_dm().get_change_points()
            filename = 'change_points_export.csv'
        elif data_type == 'events':
            data = get_dm().get_events()
            filename = 'events_export.csv'
        
        # Create CSV response
        response = jsonify(data)
        response.headers.add('Content-Disposition', f'attachment; filename={filename}')
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
