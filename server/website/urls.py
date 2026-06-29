from django.contrib import admin
from django.urls import path, re_path
from swag.views import *
from users.views import *


urlpatterns = [
    # Examples:
    re_path(r'^$', home_page, name='home_page'),
    re_path(r'^form/$', FormView, name='form_page'),
    re_path(r'^login/$', LoginView, name='form_page'),
    re_path(r'^register/$', RegistrationView, name='form_page'),
    re_path(r'^logout/$', LogoutView),
    re_path(r'^show_data/$', graph_plot, name='home_page'),
    path('admin/', admin.site.urls),
    re_path(r'^show_forecasted_smavg_data/$', forecasted_plot, name='home_page'),
    
    # 电力负荷预测仪表板
    path('dashboard/', load_forecasting_dashboard, name='forecast_dashboard'),
    
    # API 端点
    path('api/run_prediction/', run_prediction, name='api_run_prediction'),
    path('api/get_prediction_results/', get_prediction_results, name='api_get_results'),
    path('api/get_chart/', get_chart, name='api_get_chart'),
    path('api/generate_report/', generate_analysis_report, name='api_generate_report'),
    path('api/get_saved_models/', get_saved_models, name='api_get_saved_models'),
    path('api/export_report/', export_report, name='api_export_report'),
]