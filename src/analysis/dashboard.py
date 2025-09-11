# src/analysis/dashboard.py

"""
실시간 대시보드 서버
Dash와 Plotly를 사용한 웹 기반 모니터링 대시보드
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import time
from collections import deque

class DashboardServer:
    """
    실시간 대시보드 서버
    
    FinFlow-RL의 성능, 포트폴리오, 리스크 메트릭을 실시간으로 시각화
    """
    
    def __init__(self, port: int = 8050, monitor: Optional[Any] = None):
        """
        Args:
            port: 대시보드 포트
            monitor: PerformanceMonitor 인스턴스
        """
        self.port = port
        self.monitor = monitor
        self.app = dash.Dash(__name__)
        
        # 데이터 버퍼
        self.data_buffer = {
            'timestamps': deque(maxlen=1000),
            'portfolio_value': deque(maxlen=1000),
            'sharpe_ratio': deque(maxlen=1000),
            'returns': deque(maxlen=1000),
            'drawdown': deque(maxlen=1000),
            'weights': deque(maxlen=1000),
            'alerts': deque(maxlen=100)
        }
        
        # 레이아웃 설정
        self._setup_layout()
        
        # 콜백 설정
        self._setup_callbacks()
        
    def _setup_layout(self):
        """대시보드 레이아웃 설정"""
        
        self.app.layout = html.Div([
            # 헤더
            html.Div([
                html.H1("FinFlow-RL Live Dashboard", 
                       style={'text-align': 'center', 'color': '#2c3e50'}),
                html.P(f"실시간 포트폴리오 모니터링 시스템", 
                      style={'text-align': 'center', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'background': '#ecf0f1'}),
            
            # 주요 메트릭 카드
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Portfolio Value", style={'margin': '0'}),
                        html.H2(id='portfolio-value', children='$0.00',
                               style={'margin': '0', 'color': '#27ae60'}),
                        html.P(id='portfolio-change', children='0.00%',
                              style={'margin': '0', 'color': '#27ae60'})
                    ], className='metric-card', style={
                        'background': 'white',
                        'padding': '20px',
                        'border-radius': '10px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'flex': '1',
                        'margin': '10px'
                    })
                ], style={'display': 'flex', 'flex-direction': 'column'}),
                
                html.Div([
                    html.Div([
                        html.H3("Sharpe Ratio", style={'margin': '0'}),
                        html.H2(id='sharpe-ratio', children='0.00',
                               style={'margin': '0', 'color': '#3498db'}),
                        html.P("Risk-Adjusted Return", style={'margin': '0', 'color': '#95a5a6'})
                    ], className='metric-card', style={
                        'background': 'white',
                        'padding': '20px',
                        'border-radius': '10px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'flex': '1',
                        'margin': '10px'
                    })
                ], style={'display': 'flex', 'flex-direction': 'column'}),
                
                html.Div([
                    html.Div([
                        html.H3("Max Drawdown", style={'margin': '0'}),
                        html.H2(id='max-drawdown', children='0.00%',
                               style={'margin': '0', 'color': '#e74c3c'}),
                        html.P("Peak to Trough", style={'margin': '0', 'color': '#95a5a6'})
                    ], className='metric-card', style={
                        'background': 'white',
                        'padding': '20px',
                        'border-radius': '10px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'flex': '1',
                        'margin': '10px'
                    })
                ], style={'display': 'flex', 'flex-direction': 'column'}),
                
                html.Div([
                    html.Div([
                        html.H3("Daily Return", style={'margin': '0'}),
                        html.H2(id='daily-return', children='0.00%',
                               style={'margin': '0', 'color': '#f39c12'}),
                        html.P("Today's Performance", style={'margin': '0', 'color': '#95a5a6'})
                    ], className='metric-card', style={
                        'background': 'white',
                        'padding': '20px',
                        'border-radius': '10px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'flex': '1',
                        'margin': '10px'
                    })
                ], style={'display': 'flex', 'flex-direction': 'column'})
            ], style={'display': 'flex', 'padding': '20px'}),
            
            # 차트 영역
            html.Div([
                # 포트폴리오 가치 차트
                html.Div([
                    dcc.Graph(id='portfolio-chart', style={'height': '400px'})
                ], style={'padding': '20px'}),
                
                # 포트폴리오 가중치 차트
                html.Div([
                    html.Div([
                        dcc.Graph(id='weights-chart', style={'height': '400px'})
                    ], style={'flex': '1', 'padding': '10px'}),
                    html.Div([
                        dcc.Graph(id='risk-chart', style={'height': '400px'})
                    ], style={'flex': '1', 'padding': '10px'})
                ], style={'display': 'flex'}),
            ]),
            
            # 알림 패널
            html.Div([
                html.H3("Recent Alerts", style={'margin-bottom': '10px'}),
                html.Div(id='alerts-panel', style={
                    'background': '#f8f9fa',
                    'padding': '15px',
                    'border-radius': '5px',
                    'max-height': '200px',
                    'overflow-y': 'auto'
                })
            ], style={'padding': '20px'}),
            
            # 자동 새로고침
            dcc.Interval(
                id='interval-component',
                interval=1000,  # 1초마다 업데이트
                n_intervals=0
            ),
            
            # 데이터 저장소 (숨김)
            html.Div(id='hidden-data', style={'display': 'none'})
        ])
    
    def _setup_callbacks(self):
        """대시보드 콜백 설정"""
        
        @self.app.callback(
            [Output('portfolio-value', 'children'),
             Output('portfolio-change', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('max-drawdown', 'children'),
             Output('daily-return', 'children'),
             Output('portfolio-chart', 'figure'),
             Output('weights-chart', 'figure'),
             Output('risk-chart', 'figure'),
             Output('alerts-panel', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """대시보드 업데이트"""
            
            # 모니터에서 데이터 가져오기
            if self.monitor and hasattr(self.monitor, 'history') and self.monitor.history:
                latest_metrics = self.monitor.history[-1] if self.monitor.history else {}
            else:
                # 테스트용 더미 데이터
                latest_metrics = self._generate_dummy_data(n)
            
            # 메트릭 값
            portfolio_value = f"${latest_metrics.get('portfolio_value', 100000):,.2f}"
            portfolio_change = latest_metrics.get('portfolio_change', 0)
            portfolio_change_str = f"{portfolio_change:+.2f}%"
            portfolio_change_color = '#27ae60' if portfolio_change >= 0 else '#e74c3c'
            
            sharpe_ratio = f"{latest_metrics.get('sharpe_ratio', 0):.2f}"
            max_drawdown = f"{latest_metrics.get('max_drawdown', 0):.2f}%"
            daily_return = f"{latest_metrics.get('daily_return', 0):+.2f}%"
            
            # 포트폴리오 가치 차트
            portfolio_fig = self._create_portfolio_chart()
            
            # 가중치 차트
            weights_fig = self._create_weights_chart(latest_metrics.get('weights', {}))
            
            # 리스크 차트
            risk_fig = self._create_risk_chart(latest_metrics)
            
            # 알림 패널
            alerts_html = self._create_alerts_panel()
            
            return (
                portfolio_value,
                html.Span(portfolio_change_str, style={'color': portfolio_change_color}),
                sharpe_ratio,
                max_drawdown,
                daily_return,
                portfolio_fig,
                weights_fig,
                risk_fig,
                alerts_html
            )
    
    def _create_portfolio_chart(self) -> go.Figure:
        """포트폴리오 가치 차트 생성"""
        
        if self.monitor and hasattr(self.monitor, 'history'):
            history = pd.DataFrame(self.monitor.history[-100:])  # 최근 100개
            if not history.empty and 'portfolio_value' in history.columns:
                x = list(range(len(history)))
                y = history['portfolio_value'].values
            else:
                x = list(range(100))
                y = 100000 + np.cumsum(np.random.randn(100) * 1000)
        else:
            x = list(range(100))
            y = 100000 + np.cumsum(np.random.randn(100) * 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Time Steps',
            yaxis_title='Value ($)',
            template='plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='x unified'
        )
        
        return fig
    
    def _create_weights_chart(self, weights: Dict) -> go.Figure:
        """포트폴리오 가중치 차트 생성"""
        
        if weights:
            assets = list(weights.keys())
            values = list(weights.values())
        else:
            # 더미 데이터
            assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
            values = np.random.dirichlet(np.ones(5))
        
        fig = go.Figure(data=[
            go.Pie(
                labels=assets,
                values=values,
                hole=0.4,
                marker=dict(colors=px.colors.sequential.Viridis)
            )
        ])
        
        fig.update_layout(
            title='Portfolio Allocation',
            template='plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def _create_risk_chart(self, metrics: Dict) -> go.Figure:
        """리스크 메트릭 차트 생성"""
        
        categories = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate', 'Profit Factor']
        
        if metrics:
            values = [
                metrics.get('sharpe_ratio', 0),
                metrics.get('sortino_ratio', 0),
                metrics.get('calmar_ratio', 0),
                metrics.get('win_rate', 0) * 100,
                metrics.get('profit_factor', 1)
            ]
        else:
            values = [1.5, 2.0, 1.2, 55, 1.8]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['#3498db', '#27ae60', '#f39c12', '#9b59b6', '#e74c3c']
        ))
        
        fig.update_layout(
            title='Risk Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def _create_alerts_panel(self) -> html.Div:
        """알림 패널 생성"""
        
        if self.monitor and hasattr(self.monitor, 'alerts'):
            alerts = list(self.monitor.alerts)[-10:]  # 최근 10개
        else:
            # 더미 알림
            alerts = [
                {'level': 'info', 'message': 'System initialized', 'timestamp': datetime.now()},
                {'level': 'warning', 'message': 'High volatility detected', 'timestamp': datetime.now()}
            ]
        
        alert_items = []
        for alert in alerts:
            if isinstance(alert, dict):
                level = alert.get('level', 'info')
                message = alert.get('message', '')
                timestamp = alert.get('timestamp', '')
            else:
                level = getattr(alert, 'level', 'info')
                message = getattr(alert, 'message', '')
                timestamp = getattr(alert, 'timestamp', '')
            
            color_map = {
                'info': '#3498db',
                'warning': '#f39c12',
                'critical': '#e74c3c',
                'emergency': '#c0392b'
            }
            
            alert_items.append(
                html.Div([
                    html.Span(f"[{level.upper()}]", style={
                        'color': color_map.get(level, '#95a5a6'),
                        'font-weight': 'bold',
                        'margin-right': '10px'
                    }),
                    html.Span(message),
                    html.Span(f" - {timestamp}", style={
                        'color': '#95a5a6',
                        'font-size': '0.9em',
                        'margin-left': '10px'
                    })
                ], style={'padding': '5px 0', 'border-bottom': '1px solid #ecf0f1'})
            )
        
        return html.Div(alert_items)
    
    def _generate_dummy_data(self, n: int) -> Dict:
        """테스트용 더미 데이터 생성"""
        
        base_value = 100000
        value = base_value + np.sin(n * 0.1) * 5000 + np.random.randn() * 1000
        
        return {
            'portfolio_value': value,
            'portfolio_change': np.random.randn() * 2,
            'sharpe_ratio': 1.5 + np.random.randn() * 0.3,
            'max_drawdown': -abs(np.random.randn() * 5),
            'daily_return': np.random.randn() * 2,
            'weights': {
                'AAPL': 0.2 + np.random.randn() * 0.05,
                'GOOGL': 0.2 + np.random.randn() * 0.05,
                'MSFT': 0.2 + np.random.randn() * 0.05,
                'AMZN': 0.2 + np.random.randn() * 0.05,
                'META': 0.2 + np.random.randn() * 0.05
            }
        }
    
    def run(self):
        """대시보드 서버 실행"""
        self.app.run_server(
            host='0.0.0.0',
            port=self.port,
            debug=False,
            use_reloader=False
        )

if __name__ == "__main__":
    # 테스트용 실행
    dashboard = DashboardServer(port=8050)
    print(f"대시보드 서버 시작: http://localhost:8050")
    dashboard.run()