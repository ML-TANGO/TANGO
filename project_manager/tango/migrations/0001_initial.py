# Generated by Django 3.2.25 on 2024-07-04 02:30

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('targets', '__first__'),
    ]

    operations = [
        migrations.CreateModel(
            name='AuthUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128)),
                ('last_login', models.DateTimeField(blank=True, null=True)),
                ('is_superuser', models.BooleanField()),
                ('username', models.CharField(max_length=150, unique=True)),
                ('first_name', models.CharField(max_length=150)),
                ('last_name', models.CharField(max_length=150)),
                ('email', models.CharField(max_length=254)),
                ('is_staff', models.BooleanField()),
                ('is_active', models.BooleanField()),
                ('date_joined', models.DateTimeField()),
            ],
            options={
                'db_table': 'auth_user',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Oauth2ProviderAccesstoken',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('token', models.CharField(max_length=255, unique=True)),
                ('expires', models.DateTimeField()),
                ('scope', models.TextField()),
                ('created', models.DateTimeField()),
                ('updated', models.DateTimeField()),
            ],
            options={
                'db_table': 'oauth2_provider_accesstoken',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Oauth2ProviderApplication',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('client_id', models.CharField(max_length=100, unique=True)),
                ('redirect_uris', models.TextField()),
                ('client_type', models.CharField(max_length=32)),
                ('authorization_grant_type', models.CharField(max_length=32)),
                ('client_secret', models.CharField(max_length=255)),
                ('name', models.CharField(max_length=255)),
                ('skip_authorization', models.BooleanField()),
                ('created', models.DateTimeField()),
                ('updated', models.DateTimeField()),
                ('algorithm', models.CharField(max_length=5)),
            ],
            options={
                'db_table': 'oauth2_provider_application',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Oauth2ProviderIdtoken',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('jti', models.UUIDField(unique=True)),
                ('expires', models.DateTimeField()),
                ('scope', models.TextField()),
                ('created', models.DateTimeField()),
                ('updated', models.DateTimeField()),
            ],
            options={
                'db_table': 'oauth2_provider_idtoken',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Oauth2ProviderRefreshtoken',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('token', models.CharField(max_length=255)),
                ('created', models.DateTimeField()),
                ('updated', models.DateTimeField()),
                ('revoked', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'db_table': 'oauth2_provider_refreshtoken',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Anchor',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('anchor2target_ratio', models.FloatField(blank=True, null=True)),
                ('best_possible_recall', models.FloatField(blank=True, null=True)),
            ],
            options={
                'db_table': 'anchor',
            },
        ),
        migrations.CreateModel(
            name='Arguments',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('weights', models.TextField(blank=True, default='', null=True)),
                ('resume', models.BooleanField(blank=True, null=True)),
                ('cfg', models.TextField(blank=True, default='', null=True)),
                ('data', models.TextField(blank=True, default='', null=True)),
                ('hyp', models.TextField(blank=True, default='', null=True)),
                ('epochs', models.IntegerField(blank=True, null=True)),
                ('batch_size', models.IntegerField(blank=True, null=True)),
                ('total_batch_size', models.IntegerField(blank=True, null=True)),
                ('img_size', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(blank=True, null=True), blank=True, null=True, size=None)),
                ('react', models.BooleanField(blank=True, null=True)),
                ('nosave', models.BooleanField(blank=True, null=True)),
                ('notest', models.BooleanField(blank=True, null=True)),
                ('noautoanchor', models.BooleanField(blank=True, null=True)),
                ('evolve', models.BooleanField(blank=True, null=True)),
                ('cache_images', models.BooleanField(blank=True, null=True)),
                ('image_weights', models.BooleanField(blank=True, null=True)),
                ('multi_scale', models.BooleanField(blank=True, null=True)),
                ('single_cls', models.BooleanField(blank=True, null=True)),
                ('adam', models.BooleanField(blank=True, null=True)),
                ('quad', models.BooleanField(blank=True, null=True)),
                ('sync_bn', models.BooleanField(blank=True, null=True)),
                ('linear_lr', models.BooleanField(blank=True, null=True)),
                ('global_rank', models.IntegerField(blank=True, null=True)),
                ('local_rank', models.IntegerField(blank=True, null=True)),
                ('world_size', models.IntegerField(blank=True, null=True)),
                ('workers', models.IntegerField(blank=True, null=True)),
                ('bucket', models.TextField(blank=True, default='', null=True)),
                ('project', models.TextField(blank=True, default='', null=True)),
                ('name', models.TextField(blank=True, default='', null=True)),
                ('exist_ok', models.BooleanField(blank=True, null=True)),
                ('save_dir', models.TextField(blank=True, default='', null=True)),
                ('label_smoothing', models.FloatField(blank=True, null=True)),
                ('upload_dataset', models.BooleanField(blank=True, null=True)),
                ('bbox_interval', models.IntegerField(blank=True, null=True)),
                ('save_period', models.IntegerField(blank=True, null=True)),
                ('artifact_alias', models.TextField(blank=True, default='', null=True)),
                ('freeze', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(blank=True, null=True), blank=True, null=True, size=None)),
                ('metric', models.TextField(blank=True, default='', null=True)),
                ('device', models.TextField(blank=True, null=True)),
                ('entity', models.TextField(blank=True, default='', null=True)),
            ],
            options={
                'db_table': 'arguments',
            },
        ),
        migrations.CreateModel(
            name='Basemodel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('model_name', models.TextField(blank=True, null=True)),
                ('model_size', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'basemodel',
            },
        ),
        migrations.CreateModel(
            name='BatchSize',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('low', models.IntegerField(blank=True, null=True)),
                ('high', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'batch_size',
            },
        ),
        migrations.CreateModel(
            name='Hyperparameter',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('lr0', models.FloatField(blank=True, null=True)),
                ('lrf', models.FloatField(blank=True, null=True)),
                ('lrc', models.FloatField(blank=True, null=True)),
                ('momentum', models.FloatField(blank=True, null=True)),
                ('weight_decay', models.FloatField(blank=True, null=True)),
                ('warmup_epochs', models.FloatField(blank=True, null=True)),
                ('warmup_momentum', models.FloatField(blank=True, null=True)),
                ('warmup_bias_lr', models.FloatField(blank=True, null=True)),
                ('box', models.FloatField(blank=True, null=True)),
                ('cls', models.FloatField(blank=True, null=True)),
                ('cls_pw', models.FloatField(blank=True, null=True)),
                ('obj', models.FloatField(blank=True, null=True)),
                ('obj_pw', models.FloatField(blank=True, null=True)),
                ('iou_t', models.FloatField(blank=True, null=True)),
                ('anchor_t', models.FloatField(blank=True, null=True)),
                ('fl_gamma', models.FloatField(blank=True, null=True)),
                ('hsv_h', models.FloatField(blank=True, null=True)),
                ('hsv_s', models.FloatField(blank=True, null=True)),
                ('hsv_v', models.FloatField(blank=True, null=True)),
                ('degrees', models.FloatField(blank=True, null=True)),
                ('translate', models.FloatField(blank=True, null=True)),
                ('scale', models.FloatField(blank=True, null=True)),
                ('shear', models.FloatField(blank=True, null=True)),
                ('perspective', models.FloatField(blank=True, null=True)),
                ('flipud', models.FloatField(blank=True, null=True)),
                ('fliplr', models.FloatField(blank=True, null=True)),
                ('mosaic', models.FloatField(blank=True, null=True)),
                ('mixup', models.FloatField(blank=True, null=True)),
                ('copy_paste', models.FloatField(blank=True, null=True)),
                ('paste_in', models.FloatField(blank=True, null=True)),
                ('loss_ota', models.FloatField(blank=True, null=True)),
                ('label_smoothing', models.FloatField(blank=True, null=True)),
                ('momentum_group0', models.FloatField(blank=True, null=True)),
                ('lr_group0', models.FloatField(blank=True, null=True)),
                ('momentum_group1', models.FloatField(blank=True, null=True)),
                ('lr_group1', models.FloatField(blank=True, null=True)),
                ('momentum_group2', models.FloatField(blank=True, null=True)),
                ('lr_group2', models.FloatField(blank=True, null=True)),
            ],
            options={
                'db_table': 'hyperparameter',
            },
        ),
        migrations.CreateModel(
            name='ModelSummary',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('layers', models.IntegerField(blank=True, null=True)),
                ('parameters', models.BigIntegerField(blank=True, null=True)),
                ('gradients', models.BigIntegerField(blank=True, null=True)),
                ('FLOPS', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'model_summary',
            },
        ),
        migrations.CreateModel(
            name='System',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('torch', models.TextField(blank=True, null=True)),
                ('cuda', models.TextField(blank=True, null=True)),
                ('cudnn', models.TextField(blank=True, null=True)),
                ('gpus', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'system',
            },
        ),
        migrations.CreateModel(
            name='TrainDataset',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('found', models.IntegerField(blank=True, null=True)),
                ('missing', models.IntegerField(blank=True, null=True)),
                ('empty', models.IntegerField(blank=True, null=True)),
                ('corrupted', models.IntegerField(blank=True, null=True)),
                ('current', models.IntegerField(blank=True, null=True)),
                ('total', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'train_dataset',
            },
        ),
        migrations.CreateModel(
            name='TrainLossLastStep',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('project_id', models.IntegerField(default=0)),
                ('project_version', models.IntegerField(blank=True, default=0, null=True)),
                ('is_use', models.BooleanField(default=True)),
                ('epoch', models.IntegerField(blank=True, null=True)),
                ('total_epoch', models.IntegerField(blank=True, null=True)),
                ('gpu_mem', models.TextField(blank=True, null=True)),
                ('box', models.TextField(blank=True, null=True)),
                ('obj', models.TextField(blank=True, null=True)),
                ('cls', models.IntegerField(blank=True, null=True)),
                ('total', models.IntegerField(blank=True, null=True)),
                ('label', models.IntegerField(blank=True, null=True)),
                ('step', models.IntegerField(blank=True, null=True)),
                ('total_step', models.IntegerField(blank=True, null=True)),
                ('time', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'train_loss_laststep',
            },
        ),
        migrations.CreateModel(
            name='TrainLossLatest',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('epoch', models.IntegerField(blank=True, null=True)),
                ('total_epoch', models.IntegerField(blank=True, null=True)),
                ('gpu_mem', models.TextField(blank=True, null=True)),
                ('box', models.TextField(blank=True, null=True)),
                ('obj', models.TextField(blank=True, null=True)),
                ('cls', models.IntegerField(blank=True, null=True)),
                ('total', models.IntegerField(blank=True, null=True)),
                ('label', models.IntegerField(blank=True, null=True)),
                ('step', models.IntegerField(blank=True, null=True)),
                ('total_step', models.IntegerField(blank=True, null=True)),
                ('time', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'train_loss_latest',
            },
        ),
        migrations.CreateModel(
            name='TrainStart',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('status', models.TextField(blank=True, null=True)),
                ('epochs', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'train_start',
            },
        ),
        migrations.CreateModel(
            name='ValAccuracyLastStep',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('project_id', models.IntegerField(default=0)),
                ('project_version', models.IntegerField(blank=True, default=0, null=True)),
                ('is_use', models.BooleanField(default=True)),
                ('class_type', models.TextField(blank=True, null=True)),
                ('images', models.IntegerField(blank=True, null=True)),
                ('labels', models.BigIntegerField(blank=True, null=True)),
                ('P', models.IntegerField(blank=True, null=True)),
                ('R', models.IntegerField(blank=True, null=True)),
                ('mAP50', models.IntegerField(blank=True, null=True)),
                ('mAP50_95', models.IntegerField(blank=True, null=True)),
                ('step', models.IntegerField(blank=True, null=True)),
                ('total_step', models.IntegerField(blank=True, null=True)),
                ('time', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'val_accuracy_laststep',
            },
        ),
        migrations.CreateModel(
            name='ValAccuracyLatest',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('train_loss_latest_id', models.IntegerField(blank=True, null=True)),
                ('epoch', models.IntegerField(blank=True, null=True)),
                ('total_epoch', models.IntegerField(blank=True, null=True)),
                ('class_type', models.TextField(blank=True, null=True)),
                ('images', models.IntegerField(blank=True, null=True)),
                ('labels', models.BigIntegerField(blank=True, null=True)),
                ('P', models.IntegerField(blank=True, null=True)),
                ('R', models.IntegerField(blank=True, null=True)),
                ('mAP50', models.IntegerField(blank=True, null=True)),
                ('mAP50_95', models.IntegerField(blank=True, null=True)),
                ('step', models.IntegerField(blank=True, null=True)),
                ('total_step', models.IntegerField(blank=True, null=True)),
                ('time', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'val_accuracy_latest',
            },
        ),
        migrations.CreateModel(
            name='ValDataset',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('found', models.IntegerField(blank=True, null=True)),
                ('missing', models.IntegerField(blank=True, null=True)),
                ('empty', models.IntegerField(blank=True, null=True)),
                ('corrupted', models.IntegerField(blank=True, null=True)),
                ('current', models.IntegerField(blank=True, null=True)),
                ('total', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'val_dataset',
            },
        ),
        migrations.CreateModel(
            name='WorkflowOrder',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('workflow_name', models.CharField(blank=True, max_length=30, null=True)),
                ('order', models.IntegerField(blank=True, null=True)),
                ('project_id', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'workflow_order',
            },
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('project_name', models.CharField(blank=True, max_length=30, null=True)),
                ('project_description', models.CharField(blank=True, max_length=200, null=True)),
                ('create_user', models.CharField(blank=True, max_length=50, null=True)),
                ('create_date', models.CharField(blank=True, max_length=50, null=True)),
                ('project_type', models.CharField(blank=True, max_length=50, null=True)),
                ('dataset', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('version', models.IntegerField(default=0)),
                ('task_type', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('autonn_dataset_file', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('autonn_basemodel', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('nas_type', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_weight_level', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_precision_level', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_processing_lib', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_user_edit', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_input_method', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_input_data_path', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_output_method', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('deploy_input_source', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('container', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('container_status', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('last_logs_timestamp', models.FloatField(blank=True, default=0, null=True)),
                ('last_log_container', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('current_log', models.TextField(blank=True, default='', null=True)),
                ('target', models.ForeignKey(blank=True, db_column='target', null=True, on_delete=django.db.models.deletion.PROTECT, related_name='target', to='targets.target')),
            ],
            options={
                'db_table': 'project',
            },
        ),
        migrations.CreateModel(
            name='AutonnStatus',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('progress', models.IntegerField(blank=True, db_column='progress', default=0, null=True)),
                ('anchor', models.ForeignKey(db_column='anchor', on_delete=django.db.models.deletion.PROTECT, related_name='anchor', to='tango.anchor')),
                ('arguments', models.ForeignKey(db_column='arguments', on_delete=django.db.models.deletion.PROTECT, related_name='arguments', to='tango.arguments')),
                ('basemodel', models.ForeignKey(db_column='basemodel', on_delete=django.db.models.deletion.PROTECT, related_name='basemodel', to='tango.basemodel')),
                ('batch_size', models.ForeignKey(db_column='batch_size', on_delete=django.db.models.deletion.PROTECT, related_name='batch_size', to='tango.batchsize')),
                ('hyperparameter', models.ForeignKey(db_column='hyperparameter', on_delete=django.db.models.deletion.PROTECT, related_name='hyperparameter', to='tango.hyperparameter')),
                ('model_summary', models.ForeignKey(db_column='model_summary', on_delete=django.db.models.deletion.PROTECT, related_name='model_summary', to='tango.modelsummary')),
                ('project', models.ForeignKey(db_column='project', on_delete=django.db.models.deletion.PROTECT, related_name='project', to='tango.project')),
                ('system', models.ForeignKey(db_column='system', on_delete=django.db.models.deletion.PROTECT, related_name='system', to='tango.system')),
                ('train_dataset', models.ForeignKey(db_column='train_dataset', on_delete=django.db.models.deletion.PROTECT, related_name='train_dataset', to='tango.traindataset')),
                ('train_loss_latest', models.ForeignKey(db_column='train_loss_latest', on_delete=django.db.models.deletion.PROTECT, related_name='train_loss_latest', to='tango.trainlosslatest')),
                ('train_start', models.ForeignKey(db_column='train_start', on_delete=django.db.models.deletion.PROTECT, related_name='train_start', to='tango.trainstart')),
                ('val_accuracy_latest', models.ForeignKey(db_column='val_accuracy_latest', on_delete=django.db.models.deletion.PROTECT, related_name='val_accuracy_latest', to='tango.valaccuracylatest')),
                ('val_dataset', models.ForeignKey(db_column='val_dataset', on_delete=django.db.models.deletion.PROTECT, related_name='val_dataset', to='tango.valdataset')),
            ],
            options={
                'db_table': 'autonn_status',
            },
        ),
    ]
