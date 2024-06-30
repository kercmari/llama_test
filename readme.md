## RUN ENV
```
python -m venv env
env\Scripts\activate

source env/bin/activate
```

En caso de no tener permisos 

`Set-ExecutionPolicy RemoteSigned`

Instalar librerias 
`pip install -r requirements.txt `
Para compilar el codigo 
`python src/app.py`

## Documentation Config 

```json
 {
        "key_type": "string o array", //Tipo de concatenacion para la nueva data Ej "32323-5454 o  ['43434','322332']"
        "key_prop": "illness", //Variable de listado de IDs que se usara como comparacion
        "split_key": " ", //Key word para separar los IDs
        "value_prop": "", //Valor a ser concatendo o agrupado , es decir quien contiene los IDs 
        "info_key": "", //Valor del set de datos donde se saca la informacion o el contexto
        "update_key": "", //Key porpertie a modificar, es decir la que va a ser afectada con los IDs
        "data_ids_path": "listEnfermdadMedicamente.json", //nombre del archivo de la lista de Ids 
        "update_data_path": "pacientes2.json" //nombre del archivo de la data actualizar
    }
```
