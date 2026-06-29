from django.shortcuts import render, HttpResponse
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt
import datetime
from datetime import timedelta
import json
import requests
import csv
import os
import sys
import pandas as pd
from .models import CSV

# 添加项目路径到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 动态导入 myModel.lstm 模块
import importlib.util
lstm_file = os.path.join(project_root, 'myModel.lstm.py')
spec = importlib.util.spec_from_file_location("myModel_lstm", lstm_file)
myModel_lstm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(myModel_lstm)
LoadForecaster = myModel_lstm.LoadForecaster

from generate_report import build_analyst_prompt, call_qianfan_agent


# 全局变量存储预测器实例和报告内容
_forecaster_instance = None
_prediction_results = None
_last_report_content = None


def get_forecaster():
    """获取或创建预测器实例"""
    global _forecaster_instance
    if _forecaster_instance is None:
        data_file = os.path.join(
            project_root,
            'powerdemand_5min_2021_to_2024_with weather.csv'
        )
        _forecaster_instance = LoadForecaster(file_path=data_file)
    return _forecaster_instance


def home_page(request):
    """主页"""
    print("date:", datetime.date.today().day)
    day = datetime.date.today().day
    month = datetime.date.today().month
    year = datetime.date.today().year
    return render(request, "Home_page.html", {
        'Day': str(day).zfill(2),
        'Month': str(month).zfill(2),
        'Year': year
    })


def load_forecasting_dashboard(request):
    """负荷预测仪表板页面"""
    return render(request, 'forecast_dashboard.html')


def get_saved_models(request):
    """获取已保存的模型列表"""
    try:
        forecaster = get_forecaster()
        models = forecaster.list_saved_models()
        return HttpResponse(
            json.dumps({
                'status': 'success',
                'models': models
            }, ensure_ascii=False),
            content_type='application/json'
        )
    except Exception as e:
        return HttpResponse(
            json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False),
            content_type='application/json'
        )


@csrf_exempt
def run_prediction(request):
    """运行预测流程API"""
    if request.method == 'POST':
        try:
            forecaster = get_forecaster()

            # 解析请求参数
            body = json.loads(request.body) if request.body else {}
            forecast_hours = int(body.get('forecast_hours', 24))
            use_saved_model = body.get('use_saved_model', None)
            save_model = body.get('save_model', True)
            epochs = int(body.get('epochs', 20))

            # 设置训练轮数
            if not use_saved_model:
                forecaster.EPOCHS = epochs

            # 运行完整预测流程
            result = forecaster.run_full_pipeline(
                forecast_hours=forecast_hours,
                use_saved_model=use_saved_model,
                save_model=save_model
            )

            # 保存结果到全局变量
            global _prediction_results
            _prediction_results = result

            # 返回结果
            response_data = {
                'status': 'success',
                'message': '预测完成',
                'stats': result['stats'],
                'anomaly_count': len(result['anomalies']),
                'model_name': result.get('model_name'),
                'forecast_hours': result.get('forecast_hours'),
                'training_epochs': len(result.get('training_history', []))
            }

            return HttpResponse(
                json.dumps(response_data, ensure_ascii=False),
                content_type='application/json'
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return HttpResponse(
                json.dumps({
                    'status': 'error',
                    'message': str(e)
                }, ensure_ascii=False),
                content_type='application/json'
            )
    else:
        return HttpResponse(
            json.dumps({'status': 'error', 'message': '仅支持POST请求'}),
            content_type='application/json'
        )


def get_prediction_results(request):
    """获取预测结果API"""
    global _prediction_results

    if _prediction_results is None:
        return HttpResponse(
            json.dumps({
                'status': 'error',
                'message': '尚未运行预测,请先调用 /api/run_prediction/'
            }, ensure_ascii=False),
            content_type='application/json'
        )

    try:
        response_data = {
            'status': 'success',
            'stats': _prediction_results['stats'],
            'prediction_data': _prediction_results['prediction_data'],
            'anomalies': _prediction_results['anomalies'],
            'chart_base64': _prediction_results['chart_base64'],
            'model_name': _prediction_results.get('model_name'),
            'forecast_hours': _prediction_results.get('forecast_hours')
        }

        return HttpResponse(
            json.dumps(response_data, ensure_ascii=False),
            content_type='application/json'
        )

    except Exception as e:
        return HttpResponse(
            json.dumps({
                'status': 'error',
                'message': str(e)
            }, ensure_ascii=False),
            content_type='application/json'
        )


def get_chart(request):
    """获取预测图表API"""
    global _prediction_results

    if _prediction_results is None or 'chart_base64' not in _prediction_results:
        return HttpResponse(
            json.dumps({
                'status': 'error',
                'message': '图表尚未生成'
            }, ensure_ascii=False),
            content_type='application/json'
        )

    return HttpResponse(
        json.dumps({
            'status': 'success',
            'chart_base64': _prediction_results['chart_base64']
        }, ensure_ascii=False),
        content_type='application/json'
    )


@csrf_exempt
def generate_analysis_report(request):
    """生成分析简报API"""
    if request.method != 'POST':
        return HttpResponse(
            json.dumps({'status': 'error', 'message': '仅支持POST请求'}),
            content_type='application/json'
        )

    global _prediction_results
    global _last_report_content

    if _prediction_results is None:
        return HttpResponse(
            json.dumps({
                'status': 'error',
                'message': '尚未运行预测,请先调用 /api/run_prediction/'
            }, ensure_ascii=False),
            content_type='application/json'
        )

    try:
        stats = _prediction_results['stats']
        anomalies_df = pd.DataFrame(_prediction_results['anomalies'])
        prompt = build_analyst_prompt(stats, anomalies_df)

        report = call_qianfan_agent(prompt)
        _last_report_content = report

        report_data = {
            'status': 'success',
            'report': report,
            'generated_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return HttpResponse(
            json.dumps(report_data, ensure_ascii=False),
            content_type='application/json'
        )

    except Exception as e:
        return HttpResponse(
            json.dumps({
                'status': 'error',
                'message': str(e)
            }, ensure_ascii=False),
            content_type='application/json'
        )


@csrf_exempt
def export_report(request):
    """导出分析报告API"""
    if request.method != 'POST':
        return HttpResponse(
            json.dumps({'status': 'error', 'message': '仅支持POST请求'}),
            content_type='application/json'
        )

    global _prediction_results
    global _last_report_content

    if _prediction_results is None:
        return HttpResponse(
            json.dumps({
                'status': 'error',
                'message': '尚未运行预测'
            }, ensure_ascii=False),
            content_type='application/json'
        )

    try:
        body = json.loads(request.body) if request.body else {}
        export_format = body.get('format', 'md')

        forecaster = get_forecaster()
        report_content = _last_report_content or "报告尚未生成，请先点击生成简报按钮。"

        report_text, filename = forecaster.export_report(
            _prediction_results,
            report_content,
            format=export_format
        )

        # 设置响应头以触发下载
        content_types = {
            'md': 'text/markdown',
            'html': 'text/html',
            'txt': 'text/plain'
        }
        content_type = content_types.get(export_format, 'text/plain')

        response = HttpResponse(report_text, content_type=content_type)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    except Exception as e:
        return HttpResponse(
            json.dumps({
                'status': 'error',
                'message': str(e)
            }, ensure_ascii=False),
            content_type='application/json'
        )


def graph_plot(request):
    """历史数据图表绘制"""
    if request.method == 'POST':
        a = request.POST['from']
        b = request.POST['to']
        ans = a.split('-')
        bns = b.split('-')
        aa = datetime.date(int(ans[0]), int(ans[1]), int(ans[2]))
        bb = datetime.date(int(bns[0]), int(bns[1]), int(bns[2]))
        query_results = []
        last = []
        t = ['x', '00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30', '00:35', '00:40', '00:45', '00:50', '00:55',
             '01:00', '01:05', '01:10', '01:15', '01:20', '01:25', '01:30', '01:35', '01:40', '01:45', '01:50', '01:55',
             '02:00', '02:05', '02:10', '02:15', '02:20', '02:25', '02:30', '02:35', '02:40', '02:45', '02:50', '02:55',
             '03:00', '03:05', '03:10', '03:15', '03:20', '03:25', '03:30', '03:35', '03:40', '03:45', '03:50', '03:55',
             '04:00', '04:05', '04:10', '04:15', '04:20', '04:25', '04:30', '04:35', '04:40', '04:45', '04:50', '04:55',
             '05:00', '05:05', '05:10', '05:15', '05:20', '05:25', '05:30', '05:35', '05:40', '05:45', '05:50', '05:55',
             '06:00', '06:05', '06:10', '06:15', '06:20', '06:25', '06:30', '06:35', '06:40', '06:45', '06:50', '06:55',
             '07:00', '07:05', '07:10', '07:15', '07:20', '07:25', '07:30', '07:35', '07:40', '07:45', '07:50', '07:55',
             '08:00', '08:05', '08:10', '08:15', '08:20', '08:25', '08:30', '08:35', '08:40', '08:45', '08:50', '08:55',
             '09:00', '09:05', '09:10', '09:15', '09:20', '09:25', '09:30', '09:35', '09:40', '09:45', '09:50', '09:55',
             '10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35', '10:40', '10:45', '10:50', '10:55',
             '11:00', '11:05', '11:10', '11:15', '11:20', '11:25', '11:30', '11:35', '11:40', '11:45', '11:50', '11:55',
             '12:00', '12:05', '12:10', '12:15', '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55',
             '13:00', '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40', '13:45', '13:50', '13:55',
             '14:00', '14:05', '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40', '14:45', '14:50', '14:55',
             '15:00', '15:05', '15:10', '15:15', '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55',
             '16:00', '16:05', '16:10', '16:15', '16:20', '16:25', '16:30', '16:35', '16:40', '16:45', '16:50', '16:55',
             '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55',
             '18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45', '18:50', '18:55',
             '19:00', '19:05', '19:10', '19:15', '19:20', '19:25', '19:30', '19:35', '19:40', '19:45', '19:50', '19:55',
             '20:00', '20:05', '20:10', '20:15', '20:20', '20:25', '20:30', '20:35', '20:40', '20:45', '20:50', '20:55',
             '21:00', '21:05', '21:10', '21:15', '21:20', '21:25', '21:30', '21:35', '21:40', '21:45', '21:50', '21:55',
             '22:00', '22:05', '22:10', '22:15', '22:20', '22:25', '22:30', '22:35', '22:40', '22:45', '22:50', '22:55',
             '23:00', '23:05', '23:10', '23:15', '23:20', '23:25', '23:30', '23:35', '23:40', '23:45', '23:50', '23:55']

        for i in range((bb - aa).days + 1):
            query_results.append(CSV.objects.filter(date=aa + timedelta(days=i)).order_by('timestamp'))
            q = [str(aa + timedelta(days=i))]
            for x in query_results[i]:
                q.append(x.load_value)
            last.append(q)

        last.insert(0, t)
    else:
        last = None

    cont = {
        'Load': last,
    }

    return HttpResponse(json.dumps(cont), content_type='application/json')


def forecasted_plot(request):
    """预测数据图表绘制"""
    if request.method == 'POST':
        qq = request.POST['fc']
        ans = qq.split('-')
        aa = datetime.date(int(ans[0]), int(ans[1]), int(ans[2]))
        day = aa.day
        month = aa.month
        year = aa.year
        query_results = []
        l = []
        query_results.append((CSV.objects.filter(date=aa).order_by('timestamp')))
        q = [str(aa)]
        for x in query_results[0]:
            q.append(x.load_value)

        ARIMA_load = ['Forecasted with ARIMA']
        WMA_load = ['Forecasted with WMA']
        SMA_load = ['Forecasted with SMA']
        LSTM_load = ['Forecasted with LSTM']
        SES_load = ['Forecasted with SES']
        GRU_load = ['Forecasted with GRU']
        RNN_load = ['Forecasted with RNN']

        csv_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        arima_csv = pd.read_csv(os.path.join(csv_path, 'predictions/ARIMA/' + str(day).zfill(2) + "-" + str(month).zfill(2) + "-" + str(year) + '.csv'))['load'].values
        ARIMA_load.extend(list(arima_csv))

        wma_csv = pd.read_csv(os.path.join(csv_path, 'predictions/WMA/' + str(day).zfill(2) + "-" + str(month).zfill(2) + "-" + str(year) + '.csv'))['load'].values
        WMA_load.extend(list(wma_csv))

        sma_csv = pd.read_csv(os.path.join(csv_path, 'predictions/SMA/' + str(day).zfill(2) + "-" + str(month).zfill(2) + "-" + str(year) + '.csv'))['load'].values
        SMA_load.extend(list(sma_csv))

        ses_csv = pd.read_csv(os.path.join(csv_path, 'predictions/SES/' + str(day).zfill(2) + "-" + str(month).zfill(2) + "-" + str(year) + '.csv'))['load'].values
        SES_load.extend(list(ses_csv))

        lstm_csv = pd.read_csv(os.path.join(csv_path, 'predictions/LSTM/' + str(day).zfill(2) + "-" + str(month).zfill(2) + "-" + str(year) + '.csv'))['load'].values
        LSTM_load.extend(list(lstm_csv))

        gru_csv = pd.read_csv(os.path.join(csv_path, 'predictions/GRU/' + str(day).zfill(2) + "-" + str(month).zfill(2) + "-" + str(year) + '.csv'))['load'].values
        GRU_load.extend(list(gru_csv))

        rnn_csv = pd.read_csv(os.path.join(csv_path, 'predictions/RNN/' + str(day).zfill(2) + "-" + str(month).zfill(2) + "-" + str(year) + '.csv'))['load'].values
        RNN_load.extend(list(rnn_csv))

        l.append(SMA_load)
        l.append(WMA_load)
        l.append(SES_load)
        l.append(ARIMA_load)
        l.append(LSTM_load)
        l.append(GRU_load)
        l.append(RNN_load)
        l.append(q)
        redu = q.count(None)
        length = len(q) - redu

        t = ['x', '00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30', '00:35', '00:40', '00:45', '00:50', '00:55',
             '01:00', '01:05', '01:10', '01:15', '01:20', '01:25', '01:30', '01:35', '01:40', '01:45', '01:50', '01:55',
             '02:00', '02:05', '02:10', '02:15', '02:20', '02:25', '02:30', '02:35', '02:40', '02:45', '02:50', '02:55',
             '03:00', '03:05', '03:10', '03:15', '03:20', '03:25', '03:30', '03:35', '03:40', '03:45', '03:50', '03:55',
             '04:00', '04:05', '04:10', '04:15', '04:20', '04:25', '04:30', '04:35', '04:40', '04:45', '04:50', '04:55',
             '05:00', '05:05', '05:10', '05:15', '05:20', '05:25', '05:30', '05:35', '05:40', '05:45', '05:50', '05:55',
             '06:00', '06:05', '06:10', '06:15', '06:20', '06:25', '06:30', '06:35', '06:40', '06:45', '06:50', '06:55',
             '07:00', '07:05', '07:10', '07:15', '07:20', '07:25', '07:30', '07:35', '07:40', '07:45', '07:50', '07:55',
             '08:00', '08:05', '08:10', '08:15', '08:20', '08:25', '08:30', '08:35', '08:40', '08:45', '08:50', '08:55',
             '09:00', '09:05', '09:10', '09:15', '09:20', '09:25', '09:30', '09:35', '09:40', '09:45', '09:50', '09:55',
             '10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35', '10:40', '10:45', '10:50', '10:55',
             '11:00', '11:05', '11:10', '11:15', '11:20', '11:25', '11:30', '11:35', '11:40', '11:45', '11:50', '11:55',
             '12:00', '12:05', '12:10', '12:15', '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55',
             '13:00', '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40', '13:45', '13:50', '13:55',
             '14:00', '14:05', '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40', '14:45', '14:50', '14:55',
             '15:00', '15:05', '15:10', '15:15', '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55',
             '16:00', '16:05', '16:10', '16:15', '16:20', '16:25', '16:30', '16:35', '16:40', '16:45', '16:50', '16:55',
             '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55',
             '18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45', '18:50', '18:55',
             '19:00', '19:05', '19:10', '19:15', '19:20', '19:25', '19:30', '19:35', '19:40', '19:45', '19:50', '19:55',
             '20:00', '20:05', '20:10', '20:15', '20:20', '20:25', '20:30', '20:35', '20:40', '20:45', '20:50', '20:55',
             '21:00', '21:05', '21:10', '21:15', '21:20', '21:25', '21:30', '21:35', '21:40', '21:45', '21:50', '21:55',
             '22:00', '22:05', '22:10', '22:15', '22:20', '22:25', '22:30', '22:35', '22:40', '22:45', '22:50', '22:55',
             '23:00', '23:05', '23:10', '23:15', '23:20', '23:25', '23:30', '23:35', '23:40', '23:45', '23:50', '23:55']
        l.insert(0, t)
    else:
        l = None

    def mean_absolute_percentage_error(y_pred, y_true):
        try:
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except Exception as e:
            mape = e
        return mape

    def root_mean_square_error(y_pred, y_true):
        try:
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            rmse = np.sqrt((y_pred - y_true) ** 2).mean()
        except Exception as e:
            rmse = e
        return rmse

    import numpy as np

    cont = {
        'forecasted_Load': l,
        'rmseSMA': round(root_mean_square_error(l[1][1:length], l[8][1:length]), 2),
        'rmseWMA': round(root_mean_square_error(l[2][1:length], l[8][1:length]), 2),
        'rmseSES': round(root_mean_square_error(l[3][1:length], l[8][1:length]), 2),
        'rmseARIMA': round(root_mean_square_error(l[4][1:length], l[8][1:length]), 2),
        'rmseLSTM': round(root_mean_square_error(l[5][1:length], l[8][1:length]), 2),
        'rmseGRU': round(root_mean_square_error(l[6][1:length], l[8][1:length]), 2),
        'rmseRNN': round(root_mean_square_error(l[7][1:length], l[8][1:length]), 2),
        'mapeSMA': round(mean_absolute_percentage_error(l[1][1:length], l[8][1:length]), 2),
        'mapeWMA': round(mean_absolute_percentage_error(l[2][1:length], l[8][1:length]), 2),
        'mapeSES': round(mean_absolute_percentage_error(l[3][1:length], l[8][1:length]), 2),
        'mapeARIMA': round(mean_absolute_percentage_error(l[4][1:length], l[8][1:length]), 2),
        'mapeLSTM': round(mean_absolute_percentage_error(l[5][1:length], l[8][1:length]), 2),
        'mapeGRU': round(mean_absolute_percentage_error(l[6][1:length], l[8][1:length]), 2),
        'mapeRNN': round(mean_absolute_percentage_error(l[7][1:length], l[8][1:length]), 2),
    }

    return HttpResponse(json.dumps(cont), content_type='application/json')