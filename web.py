#!/usr/bin/env python3
"""Веб-интерфейс: чат с агентом и наглядная раскладка памяти по слоям.

Запуск:
    python web.py
Затем открыть http://127.0.0.1:5000

Главная мысль страницы — не чат, а то, что справа от него: слои памяти со своим
содержимым и профиль, который пользователь правит прямо здесь. Изменил
предпочтение — следующий ответ собирается уже по-новому, и это видно в разборе
промпта и в проверке ответа на соответствие профилю.

Как и cli.py, этот файл — только интерфейс: он вызывает публичные методы агента
и рисует результат.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request

from agent import AgentError, MemoryAgent
from agent import catalog, interview, invariants as inv, preferences
from agent.memory.manager import ASK, AUTO, LONG, OFF, SHORT, WORKING
from agent.memory.working import STAGE_LABELS, STAGES
from agent.scenarios import Scenario, ScenarioError, Step
from agent import transitions as tr
from agent.transitions import ПереходОтклонён
from agent import mcp

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("web")

app = Flask(__name__)
agent = MemoryAgent()

# Режим, которого у самого агента нет: запускать ли сценарий по триггеру. В
# консоли это ключ «--без-сценариев», здесь — переключатель на странице.
_режимы = {"автозапуск сценариев": True}

# Прогон сценария идёт в отдельном потоке, а страница спрашивает о ходе дела.
# Иначе один HTTP-запрос держится всё время прогона — а это от полутора минут до
# нескольких, если шаги упираются в минутный лимит провайдера или уходят на
# повтор. Пользователь в это время видит неподвижный экран и не может отличить
# работу от зависания; именно так и выглядела первая версия.
_прогон: dict = {}
_замок = threading.Lock()


def _прогон_идёт() -> bool:
    return bool(_прогон) and not _прогон.get("готово")

# Имена слоёв в том виде, в каком их присылает страница.
ИМЕНА_СЛОЁВ = {"кратко": SHORT, "рабочая": WORKING, "долго": LONG}


def _применить(данные: dict) -> None:
    """Приводит агента в состояние, выбранное на странице.

    Страница живёт дольше запроса, поэтому набор слоёв, режим маршрутизатора и
    текущая задача присылаются с каждым обращением: так после перезапуска
    сервера страница не окажется рассинхронизирована с агентом.
    """
    слои = данные.get("layers")
    if isinstance(слои, list):
        agent.set_layers({ИМЕНА_СЛОЁВ[с] for с in слои if с in ИМЕНА_СЛОЁВ})

    режим = (данные.get("router") or "").strip()
    if режим in (AUTO, ASK, OFF):
        agent.memory.router_mode = режим

    модель = (данные.get("model") or "").strip()
    agent.model_key = модель if модель in catalog.MODELS else ""

    пользователь = (данные.get("user") or "").strip()
    if пользователь and пользователь != agent.user_id:
        _сменить_пользователя(пользователь)

    if "tools" in данные:
        _переключить_инструменты(данные.get("tools"))

    задача = (данные.get("task") or "").strip()
    if задача and (agent.task is None or agent.task.task_id != задача):
        agent.use_task(задача)
    elif not задача and agent.task is not None:
        agent.drop_task()


def _переключить_инструменты(значение) -> None:
    """Включает или выключает инструменты MCP по состоянию переключателя.

    Страница присылает список серверов, когда переключатель включён, и пустоту,
    когда выключен. Соединения при выключении закрываются: stdio-сервер — это
    живой процесс, и держать его «на всякий случай» незачем.
    """
    имена = [str(и).strip() for и in значение if str(и).strip()] if isinstance(значение, list) else []
    if not имена:
        if agent.toolbox is not None:
            agent.toolbox.close()
            agent.toolbox = None
        return
    if agent.toolbox is not None and agent.toolbox.имена == [и for и in имена if и != mcp.ВСЕ]:
        return
    if agent.toolbox is not None:
        agent.toolbox.close()
    agent.toolbox = mcp.Toolbox(имена, agent.mcp_config)


def _поля_профиля() -> list[dict]:
    """Описание настраиваемых полей — из него страница рисует форму профиля."""
    return [
        {
            "раздел": поле.section, "ключ": поле.key, "название": поле.label,
            "тип": поле.kind, "варианты": list(поле.options),
            "подсказка": поле.hint,
            "проверяется": "жёстко" if (поле.check and поле.hard)
                           else ("мягко" if поле.check else ""),
        }
        for поле in preferences.FIELDS
    ]


def _состояние() -> dict:
    """Всё, что странице нужно, чтобы нарисовать текущее положение дел."""
    return {
        "info": agent.info(),
        "task_state": agent.task_state(),
        "invariants": [
            {
                "код": и.код, "правило": и.правило, "вид": и.вид, "тип": и.тип,
                "значения": и.значения, "почему": и.почему, "вместо": и.вместо,
                "источник": и.источник, "уровень": и.уровень,
                "проверка": inv.ТИПЫ_СЛОВАМИ[и.тип], "жёсткий": и.жёсткий,
            }
            for и in agent.invariants()
        ],
        "invariant_kinds": [{"вид": в, "словами": inv.ВИДЫ_СЛОВАМИ[в]} for в in inv.ВИДЫ],
        "decisions_for_promote": [
            {"id": з.get("id"), "заголовок": з.get("заголовок", "")}
            for з in agent.memory.long.decisions.all()
        ],
        "stats": agent.stats(),
        "files": agent.files(),
        "tasks": agent.tasks(),
        "profile": agent.profile(),
        "profile_fields": _поля_профиля(),
        "profile_summary": preferences.summary(agent.profile()),
        "needs_setup": agent.needs_setup(),
        "templates": interview.describe_templates(),
        "setup_questions": agent.setup_questions(),
        "roles": list(catalog.ROLES),
        "stages": list(STAGES),
        "conditions": [
            {"код": у.код, "откуда": у.откуда, "куда": у.куда, "что": у.что,
             "значение": у.значение, "правило": у.правило, "почему": у.почему,
             "вместо": у.вместо, "уровень": у.уровень,
             "проверка": tr.ПРОВЕРКИ_СЛОВАМИ.get(у.что, у.что)}
            for у in agent.conditions()
        ],
        "condition_checks": [
            {"что": п, "словами": tr.ПРОВЕРКИ_СЛОВАМИ[п]} for п in tr.ПРОВЕРКИ
        ],
        "stage_labels": dict(STAGE_LABELS),
        "modes": {
            "auto_scenarios": _режимы["автозапуск сценариев"],
            "follow_profile": agent.follow_profile,
            "soft_check": agent.soft_check,
            "judge_semantic": agent.judge_semantic,
            "require_self_report": agent.require_self_report,
        },
        "scenarios": [
            {
                "имя": с.имя, "описание": с.описание, "триггеры": с.триггеры,
                "шаги": [
                    {"агент": ш.агент, "стадия": ш.стадия, "модель": ш.model_key,
                     "вход": ш.вход, "задача": ш.задача}
                    for ш in с.шаги
                ],
            }
            for с in agent.scenarios()
        ],
        # Планировщик: только счётчики и пульс работника. Сами задания и сводки
        # страница берёт отдельной ручкой — они обновляются по таймеру, а общее
        # состояние перечитывается после каждого действия, и таскать их дважды
        # незачем.
        "scheduler": agent.состояние_планировщика(),
        "pipelines": agent.состояние_конвейеров(),
        "knowledge": agent.memory.long.knowledge.all(),
        "decisions": agent.memory.long.decisions.all()[-10:],
        "dialog": agent.memory.short.all(agent.session)[-20:],
        "journal": agent.journal(20),
    }


def _сменить_пользователя(user_id: str) -> None:
    """Пересоздаёт агента под другого пользователя.

    Профиль, сценарии и знания привязаны к user_id, а агент держит открытыми
    файлы одного из них. Подменять путь на лету значило бы оставить половину
    ссылок на прежнего пользователя, поэтому агент создаётся заново.
    """
    global agent
    прежний = agent
    agent = MemoryAgent(
        user_id=user_id, session=прежний.session, layers=set(прежний.layers),
        router_mode=прежний.memory.router_mode, model_key=прежний.model_key,
        soft_check=прежний.soft_check, follow_profile=прежний.follow_profile,
    )
    прежний.close()


@app.after_request
def _не_кэшировать(ответ):
    """Страница меняется вместе с кодом, и старая в кэше выглядит как поломка.

    Именно так и вышло: сервер перезапустили, а браузер показывал прежнюю
    страницу — со старым, синхронным запуском сценария.
    """
    ответ.headers["Cache-Control"] = "no-store, must-revalidate"
    return ответ


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        info=agent.info(),
        models=catalog.describe(),
        roles=catalog.describe_roles(),
        stages=list(STAGES),
        users=list(interview.TEMPLATES),
    )


@app.get("/api/state")
def api_state():
    """Состояние всех слоёв — этим страница и оживает при открытии."""
    return jsonify(_состояние())


@app.post("/api/ask")
def api_ask():
    if _прогон_идёт():
        return jsonify({"error": "Сейчас идёт сценарий — дождитесь его конца."}), 409
    данные = request.get_json(silent=True) or {}
    вопрос = (данные.get("question") or "").strip()
    if not вопрос:
        return jsonify({"error": "Пустой вопрос."}), 400
    try:
        _применить(данные)
        ответ = agent.ask(вопрос)
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"answer": ответ.to_dict(), "state": _состояние()})


@app.post("/api/plan")
def api_plan():
    данные = request.get_json(silent=True) or {}
    try:
        _применить(данные)
        ответ = agent.plan()
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"answer": ответ.to_dict(), "state": _состояние()})


@app.post("/api/task")
def api_task():
    """Заводит задачу, берёт существующую, меняет стадию, ставит на паузу."""
    данные = request.get_json(silent=True) or {}
    действие = (данные.get("action") or "").strip()

    # Пауза — единственное действие, разрешённое во время прогона, и это не
    # поблажка, а смысл кнопки: остановить то, что идёт прямо сейчас. Флаг
    # ставится на том же состоянии задачи, с которым работает исполнитель, и
    # тот увидит его перед следующим шагом.
    if действие == "пауза":
        try:
            задача = agent.pause_task()
        except AgentError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"paused": задача.состояние_словами, "state": _состояние()})

    if _прогон_идёт():
        return jsonify({"error": "Сейчас идёт сценарий: он ведёт свою задачу."}), 409
    try:
        if действие == "создать":
            agent.start_task((данные.get("task_id") or "").strip(),
                             (данные.get("title") or "").strip())
        elif действие == "взять":
            agent.use_task((данные.get("task_id") or "").strip())
        elif действие == "стадия":
            agent.transition((данные.get("stage") or "").strip(),
                             (данные.get("note") or "").strip())
        elif действие == "закрыть-шаг":
            agent.close_step((данные.get("value") or "").strip())
        elif действие == "утвердить-план":
            подпись = agent.approve_plan((данные.get("кем") or "").strip(),
                                         (данные.get("пояснение") or "").strip())
            return jsonify({"approved": подпись, "state": _состояние()})
        elif действие == "снять-утверждение":
            agent.unapprove_plan((данные.get("почему") or "").strip())
        elif действие == "валидация":
            отчёт = agent.validate_task(ревизор=bool(данные.get("ревизор", True)))
            return jsonify({"report": отчёт, "state": _состояние()})
        elif действие == "шаг":
            agent.remember_step((данные.get("key") or "").strip(),
                                (данные.get("value") or "").strip())
        elif действие == "завершить":
            запись = agent.finish_task((данные.get("note") or "").strip())
            return jsonify({"decision": запись, "state": _состояние()})
        elif действие == "продолжить":
            agent.continue_task()
        elif действие == "ответ":
            agent.answer_task((данные.get("text") or "").strip())
        elif действие == "отпустить":
            agent.drop_task()
        else:
            return jsonify({"error": f"Неизвестное действие «{действие}»."}), 400
    except ПереходОтклонён as отклонено:
        # Отказ в переходе — не ошибка интерфейса, а содержательный ответ, и
        # страница показывает его так же, как отказ по инварианту: правилом,
        # обоснованием и тем, что нужно сделать, чтобы переход открылся.
        return jsonify({"error": str(отклонено), "refusal": отклонено.отказ.to_dict(),
                        "state": _состояние()}), 400
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"state": _состояние()})


@app.post("/api/condition")
def api_condition():
    """Личные условия перехода: добавить своё, снять своё. Базовые неподвластны."""
    данные = request.get_json(silent=True) or {}
    действие = (данные.get("action") or "добавить").strip()
    try:
        if действие == "удалить":
            код = (данные.get("код") or "").strip()
            if not agent.remove_condition(код):
                return jsonify({"error": f"Условия «{код}» нет среди личных."}), 400
        else:
            условие = tr.Условие(
                код=(данные.get("код") or "").strip(),
                откуда=(данные.get("откуда") or tr.ЛЮБАЯ).strip(),
                куда=(данные.get("куда") or tr.ЛЮБАЯ).strip(),
                что=(данные.get("что") or tr.ЕСТЬ_В_СОБРАННОМ).strip(),
                значение=(данные.get("значение") or "").strip(),
                правило=(данные.get("правило") or "").strip(),
                почему=(данные.get("почему") or "").strip(),
                вместо=(данные.get("вместо") or "").strip(),
            )
            agent.add_condition(условие)
    except (AgentError, tr.TransitionConfigError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"state": _состояние()})


@app.post("/api/remember")
def api_remember():
    """Явная запись в выбранный слой — та самая «ручная» маршрутизация."""
    данные = request.get_json(silent=True) or {}
    слой = (данные.get("target") or "").strip()
    текст = (данные.get("value") or "").strip()
    if not текст:
        return jsonify({"error": "Нечего запоминать."}), 400
    try:
        запись = agent.remember(слой, текст,
                                key=(данные.get("key") or "").strip(),
                                section=(данные.get("section") or "").strip())
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"entry": запись, "state": _состояние()})


@app.post("/api/user")
def api_user():
    """Переключение пользователя: другой профиль, другие сценарии, другие знания."""
    if _прогон_идёт():
        return jsonify({"error": "Сейчас идёт сценарий — менять пользователя нельзя."}), 409
    данные = request.get_json(silent=True) or {}
    кто = (данные.get("user") or "").strip()
    if not кто:
        return jsonify({"error": "Не указан пользователь."}), 400
    try:
        _сменить_пользователя(кто)
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"state": _состояние()})


@app.post("/api/profile")
def api_profile():
    """Правка профиля: одно предпочтение, готовая заготовка или весь мастер."""
    данные = request.get_json(silent=True) or {}
    действие = (данные.get("action") or "").strip()
    try:
        if действие == "настройка":
            agent.set_preference((данные.get("section") or "").strip(),
                                 (данные.get("key") or "").strip(),
                                 данные.get("value"))
        elif действие == "заготовка":
            agent.use_template((данные.get("template") or "").strip())
        elif действие == "мастер":
            agent.setup(данные.get("answers") or {})
        else:
            return jsonify({"error": f"Неизвестное действие «{действие}»."}), 400
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"state": _состояние()})


@app.post("/api/scenario")
def api_scenario():
    """Запускает сценарий в отдельном потоке и сразу отдаёт страницу обратно.

    Ход прогона страница забирает через /api/scenario/status: шаги появляются по
    мере готовности, а не все сразу в конце.
    """
    данные = request.get_json(silent=True) or {}
    запрос = (данные.get("query") or "").strip()
    if not запрос:
        return jsonify({"error": "Сценарию нужен запрос."}), 400
    if not _замок.acquire(blocking=False):
        return jsonify({"error": "Сценарий уже идёт. Дождитесь конца или перезагрузите страницу."}), 409

    try:
        _применить(данные)
        имя = (данные.get("name") or "").strip()
        сценарий = agent.memory.long.scenarios.get(имя) if имя else agent.match_scenario(запрос)
        if сценарий is None:
            _замок.release()
            return jsonify({"error": f"Нет сценария «{имя}»."}), 400
    except AgentError as exc:
        _замок.release()
        return jsonify({"error": str(exc)}), 400

    по_шагам = bool(данные.get("по_шагам"))
    return _запустить(
        сценарий.имя, len(сценарий.шаги),
        lambda перед, после: agent.runner.run(сценарий, запрос, on_step=перед,
                                              on_result=после, по_шагам=по_шагам),
    )


@app.post("/api/scenario/resume")
def api_scenario_resume():
    """Продолжает отложенную задачу с того шага, на котором она стоит.

    Ничего не переспрашивает: исходный запрос, результаты пройденных шагов и
    ответы человека уже лежат в состоянии задачи на диске.
    """
    данные = request.get_json(silent=True) or {}
    if not _замок.acquire(blocking=False):
        return jsonify({"error": "Сценарий уже идёт."}), 409

    task_id = (данные.get("task_id") or "").strip()
    ответ = (данные.get("answer") or "").strip()
    по_шагам = bool(данные.get("по_шагам"))
    try:
        # Выбранные на странице модель, слои и режим маршрутизатора нужно
        # применить и здесь. Без этого «Продолжить» шло на модели по роли, хотя
        # в шапке выбрана другая: после перезапуска сервера агент ничего не знал
        # о выборе человека, и шаги уходили не туда, куда он просил.
        _применить(данные)
        состояние = agent.use_task(task_id) if task_id else agent.task
        if состояние is None:
            _замок.release()
            return jsonify({"error": "Не указано, какую задачу продолжать."}), 400
        имя = состояние.сценарий or "задача"
        всего = состояние.шагов
    except AgentError as exc:
        _замок.release()
        return jsonify({"error": str(exc)}), 400

    return _запустить(
        имя, всего,
        lambda перед, после: agent.resume_scenario(
            состояние.task_id, ответ=ответ, on_step=перед, on_result=после,
            по_шагам=по_шагам),
        уже_занят=True,
    )


def _запустить(имя: str, всего: int, вызов, уже_занят: bool = False):
    """Общий запуск фонового прогона: и для нового сценария, и для продолжения."""
    _прогон.clear()
    _прогон.update({
        "id": uuid.uuid4().hex[:12],
        "сценарий": имя,
        "всего": всего,
        "шаги": [],
        "текущий": None,
        "готово": False,
        "ошибка": "",
        "решение": None,
        "на_паузе": False,
        "ожидание": "",
        "ожидание_текст": "",
        "причина_паузы": "",
    })
    поток = threading.Thread(target=_прогнать, args=(вызов,), daemon=True)
    поток.start()
    return jsonify({"run_id": _прогон["id"], "сценарий": имя, "всего": всего})


def _прогнать(вызов) -> None:
    """Тело фонового прогона. Ошибка не теряется — она ложится в состояние."""
    try:
        def перед(шаг, номер, всего):
            _прогон["текущий"] = {"агент": шаг.агент, "стадия": шаг.стадия,
                                  "модель": agent.runner.model_for(шаг),
                                  "номер": номер, "всего": всего}

        def после(результат, номер, всего):
            # Номер шага — тот, что у задачи, а не порядковый в этом прогоне.
            # После «Продолжить» прогон начинается со второго-третьего шага, и
            # без этого страница нумеровала их заново: «1/5 · архитектор» сразу
            # после «1/5 · аналитик» из прошлого захода.
            запись = результат.to_dict()
            запись["номер"] = номер
            запись["всего"] = всего
            _прогон["шаги"].append(запись)
            _прогон["текущий"] = None

        итог = вызов(перед, после)
        _прогон["решение"] = итог.решение
        _прогон["токенов"] = итог.токенов
        _прогон["стоимость"] = итог.стоимость
        # Остановка — не ошибка, а штатный исход: сценарий встал и ждёт человека.
        _прогон["на_паузе"] = итог.на_паузе
        _прогон["причина_паузы"] = итог.причина_паузы
        _прогон["ожидание"] = итог.ожидание
        _прогон["ожидание_текст"] = итог.ожидание_текст
    except Exception as exc:                      # включая AgentError и сбои сети
        log.exception("Сценарий прервался")
        _прогон["ошибка"] = str(exc)
    finally:
        _прогон["текущий"] = None
        _прогон["готово"] = True
        _замок.release()


@app.get("/api/scenario/status")
def api_scenario_status():
    """Что успел сделать идущий сценарий. Страница спрашивает это раз в секунду."""
    if not _прогон:
        return jsonify({"run": None})
    ответ = dict(_прогон)
    # Состояние памяти отдаём только в конце: пересобирать его на каждый опрос
    # значит читать все файлы слоёв несколько раз в секунду без всякой пользы.
    ответ["state"] = _состояние() if _прогон.get("готово") else None
    return jsonify({"run": ответ})


@app.post("/api/match")
def api_match():
    """Сработает ли на этом запросе сценарий — страница спрашивает до отправки.

    Решение принимается здесь, а не на странице: то же самое решает консоль, и
    правило должно быть одно на все интерфейсы.
    """
    данные = request.get_json(silent=True) or {}
    if not _режимы["автозапуск сценариев"]:
        return jsonify({"matched": None})
    сценарий = agent.match_scenario((данные.get("query") or "").strip())
    if сценарий is None:
        return jsonify({"matched": None})
    return jsonify({"matched": {"имя": сценарий.имя,
                                "шаги": [ш.агент for ш in сценарий.шаги]}})


@app.post("/api/settings")
def api_settings():
    """Переключатели режимов: то же, что ключи командной строки."""
    данные = request.get_json(silent=True) or {}
    if "auto_scenarios" in данные:
        _режимы["автозапуск сценариев"] = bool(данные["auto_scenarios"])
    if "follow_profile" in данные:
        agent.follow_profile = bool(данные["follow_profile"])
    if "soft_check" in данные:
        agent.soft_check = bool(данные["soft_check"])
    if "judge_semantic" in данные:
        agent.judge_semantic = bool(данные["judge_semantic"])
    if "require_self_report" in данные:
        agent.require_self_report = bool(данные["require_self_report"])
    return jsonify({"state": _состояние()})


@app.post("/api/invariant")
def api_invariant():
    """Заводит, снимает инвариант или возводит в него решение из журнала.

    Инварианты правятся со страницы наравне с прочим: решения о том, что
    запрещено, принимает пользователь сервиса, а не тот, кто правит JSON руками.
    """
    данные = request.get_json(silent=True) or {}
    действие = (данные.get("action") or "").strip()
    try:
        if действие == "завести":
            слова = [с.strip() for с in (данные.get("значения") or []) if с.strip()]
            инвариант = inv.Invariant(
                код=(данные.get("код") or "").strip(),
                правило=(данные.get("правило") or "").strip(),
                вид=(данные.get("вид") or inv.РЕШЕНИЕ).strip(),
                тип=inv.ЗАПРЕТ_СЛОВ if слова else inv.СМЫСЛОВОЙ,
                значения=слова,
                почему=(данные.get("почему") or "").strip(),
                вместо=(данные.get("вместо") or "").strip(),
            )
            agent.add_invariant(инвариант, личный=bool(данные.get("личный")))
        elif действие == "снять":
            if not agent.remove_invariant((данные.get("код") or "").strip()):
                return jsonify({"error": "Такого инварианта нет."}), 404
        elif действие == "возвести":
            agent.promote_decision(int(данные.get("решение") or 0))
        else:
            return jsonify({"error": f"Неизвестное действие «{действие}»."}), 400
    except (AgentError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"state": _состояние()})


@app.post("/api/scenario/save")
def api_scenario_save():
    """Создание и правка сценария целиком.

    Сценарий приходит со страницы как есть и проверяется теми же правилами, что
    и заведённый из кода: имена шагов не повторяются, роли и стадии существуют,
    маршрут проходит по разрешённым переходам. Отказ возвращается текстом —
    страница показывает его пользователю, а не молча теряет правку.
    """
    данные = request.get_json(silent=True) or {}
    try:
        шаги = [
            Step(
                агент=(ш.get("агент") or "").strip(),
                задача=(ш.get("задача") or "").strip(),
                роль=(ш.get("роль") or "исполнение").strip(),
                стадия=(ш.get("стадия") or "execution").strip(),
                вход=[в for в in (ш.get("вход") or ["запрос"]) if в],
                описание=(ш.get("описание") or "").strip(),
            )
            for ш in (данные.get("шаги") or [])
        ]
        сценарий = Scenario(
            имя=(данные.get("имя") or "").strip(),
            описание=(данные.get("описание") or "").strip(),
            триггеры=[т.strip() for т in (данные.get("триггеры") or []) if т.strip()],
            шаги=шаги,
        )
        прежнее = (данные.get("прежнее_имя") or "").strip()
        if прежнее and прежнее.lower() != сценарий.имя.lower():
            agent.remove_scenario(прежнее)
        agent.add_scenario(сценарий)
    except (AgentError, ScenarioError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"state": _состояние()})


@app.post("/api/scenario/delete")
def api_scenario_delete():
    данные = request.get_json(silent=True) or {}
    имя = (данные.get("name") or "").strip()
    if not agent.remove_scenario(имя):
        return jsonify({"error": f"Сценария «{имя}» нет."}), 404
    return jsonify({"state": _состояние()})


@app.get("/api/verify")
def api_verify():
    """Проверка каталога: отвечают ли сегодня модели, на которые он указывает."""
    return jsonify({"models": catalog.verify()})


@app.post("/api/forget")
def api_forget():
    """Очищает краткосрочную память. Остальные слои не трогает — это разные слои."""
    стёрто = agent.memory.short.clear(agent.session)
    return jsonify({"cleared": стёрто, "state": _состояние()})


# --- MCP ---------------------------------------------------------------------------
# Агент в этих ручках не участвует: серверы и их инструменты — отдельный слой, и
# список получают у самих серверов. Файл серверов перечитывается на каждый
# запрос: его правят руками, и перезапускать ради этого страницу незачем.

@app.get("/api/mcp")
def api_mcp():
    """Серверы из mcp-servers.json — без подключения, мгновенно."""
    try:
        реестр = mcp.Registry()
    except mcp.MCPConfigError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"path": реестр.path, "servers": [с.to_dict() for с in реестр.servers]})


@app.post("/api/mcp/inspect")
def api_mcp_inspect():
    """Подключиться к серверу (или ко всем) и получить список инструментов.

    Сбой соединения — не ошибка запроса: он приходит внутри осмотра, с этапом,
    причиной и подсказкой, и страница показывает его у своего сервера.
    """
    данные = request.get_json(silent=True) or {}
    имя = (данные.get("server") or "").strip()
    try:
        реестр = mcp.Registry()
        осмотры = реестр.inspect([имя] if имя else None)
    except mcp.MCPConfigError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "path": реестр.path,
        "inspections": [о.to_dict() for о in осмотры],
        "summary": mcp.summary(осмотры),
    })


@app.post("/api/tools")
def api_tools():
    """Инструменты, которые получит модель, и во что обходится их список."""
    данные = request.get_json(silent=True) or {}
    try:
        _переключить_инструменты(данные.get("tools"))
        if agent.toolbox is None:
            return jsonify({"enabled": False, "tools": [], "summary": {}})
        сводка = agent.подключить_инструменты()
        инструменты = [и.to_dict() for и in agent.toolbox.инструменты]
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"enabled": True, "tools": инструменты, "summary": сводка,
                    "confirm_writes": agent.confirm_writes})


@app.post("/api/mcp/call")
def api_mcp_call():
    """Вызов инструмента человеком со страницы — без модели и без заявки."""
    данные = request.get_json(silent=True) or {}
    имя = (данные.get("tool") or "").strip()
    аргументы = данные.get("args")
    if not имя:
        return jsonify({"error": "Не указан инструмент."}), 400
    if аргументы is not None and not isinstance(аргументы, dict):
        return jsonify({"error": "Аргументы должны быть объектом JSON."}), 400
    try:
        результат = agent.call_tool(имя, аргументы or {})
    except AgentError as exc:
        # Нет такого инструмента, сервер не поднялся, фильтр закрыл — всё это
        # ошибка запроса, а не поломка: страница показывает её текстом.
        return jsonify({"error": str(exc)}), 400
    return jsonify({"result": результат.to_dict()})


@app.get("/api/calls")
def api_calls():
    """Заявки на изменяющие вызовы, ждущие решения человека."""
    return jsonify({"calls": agent.pending_calls()})


@app.post("/api/calls")
def api_calls_resolve():
    """Подтвердить заявку (и выполнить вызов) или отклонить её."""
    данные = request.get_json(silent=True) or {}
    try:
        номер = int(данные.get("номер") or 0)
    except (TypeError, ValueError):
        номер = 0
    действие = (данные.get("действие") or "").strip()
    if not номер:
        return jsonify({"error": "Не указан номер заявки."}), 400
    try:
        if действие == "отклонить":
            решена = agent.reject_call(номер, (данные.get("почему") or "").strip())
            return jsonify({"rejected": решена, "calls": agent.pending_calls(),
                            "state": _состояние()})
        if действие == "подтвердить":
            результат = agent.confirm_call(номер)
            return jsonify({"result": результат.to_dict(), "calls": agent.pending_calls(),
                            "state": _состояние()})
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"error": "Действие: «подтвердить» или «отклонить»."}), 400


@app.get("/api/scheduler")
def api_scheduler():
    """Всё, что показывает панель планировщика: расписание, сводка, напоминания.

    Одной ручкой, а не четырьмя: панель обновляется целиком и по таймеру, и
    четыре запроса вместо одного дали бы четыре разных момента времени —
    задания из одного, пульс работника из другого.
    """
    try:
        состояние = agent.состояние_планировщика()
        return jsonify({
            "state": состояние,
            "jobs": agent.задания(только_активные=False, сколько=30),
            "digest": agent.последняя_сводка(),
            "reminders": agent.напоминания(сколько=10),
            "runs": agent.запуски(сколько=10),
        })
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/scheduler/job")
def api_scheduler_job():
    """Поставить задание в расписание или снять его — через MCP-сервер планировщика."""
    данные = request.get_json(silent=True) or {}
    _применить(данные)
    действие = (данные.get("действие") or "запланировать").strip()
    try:
        if действие == "отменить":
            номер = int(данные.get("номер") or 0)
            if not номер:
                return jsonify({"error": "Не указан номер задания."}), 400
            итог = agent.отменить_задание(номер)
        else:
            инструмент = (данные.get("инструмент") or "").strip()
            когда = (данные.get("когда") or "").strip()
            аргументы = данные.get("аргументы") or {}
            if not инструмент or not когда:
                return jsonify({"error": "Нужны инструмент и расписание."}), 400
            if not isinstance(аргументы, dict):
                return jsonify({"error": "Аргументы должны быть объектом JSON."}), 400
            итог = agent.запланировать(инструмент, когда, аргументы,
                                       (данные.get("зачем") or "").strip())
    except (AgentError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"result": итог, "jobs": agent.задания(только_активные=False, сколько=30),
                    "state": agent.состояние_планировщика()})


@app.post("/api/scheduler/digest")
def api_scheduler_digest():
    """Собрать сводку прямо сейчас — тем же инструментом, что и по расписанию."""
    данные = request.get_json(silent=True) or {}
    _применить(данные)
    за = (данные.get("за") or "сутки").strip()
    try:
        итог = agent.собрать_сводку(за)
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"result": итог, "digest": agent.последняя_сводка(),
                    "state": agent.состояние_планировщика()})


@app.get("/api/pipelines")
def api_pipelines():
    """Панель конвейеров: какие цепочки описаны и что уже выполнялось.

    Одной ручкой по той же причине, что и у планировщика: панель обновляется
    целиком, и два запроса дали бы два разных момента времени.
    """
    try:
        return jsonify({
            "state": agent.состояние_конвейеров(),
            "runs": agent.прогоны(сколько=10),
            "last": agent.последний_прогон(),
        })
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/pipeline/run")
def api_pipeline_run():
    """Выполнить цепочку целиком — тем же инструментом, которым её зовёт модель."""
    данные = request.get_json(silent=True) or {}
    _применить(данные)
    имя = (данные.get("конвейер") or "").strip()
    вход = данные.get("вход") or {}
    if not имя:
        return jsonify({"error": "Не сказано, какой конвейер запускать."}), 400
    if not isinstance(вход, dict):
        return jsonify({"error": "Вход конвейера — объект JSON."}), 400
    try:
        протокол = agent.запустить_конвейер(имя, вход, (данные.get("зачем") or "").strip())
    except (AgentError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"result": протокол, "last": agent.последний_прогон(),
                    "runs": agent.прогоны(сколько=10),
                    "state": agent.состояние_конвейеров()})


@app.get("/api/pipeline/run/<int:number>")
def api_pipeline_run_one(number: int):
    """Протокол одного прогона: вход и выход каждого шага целиком."""
    try:
        прогон = agent.прогон_конвейера(number)
    except AgentError as exc:
        return jsonify({"error": str(exc)}), 400
    if прогон is None:
        return jsonify({"error": f"Прогона №{number} нет."}), 404
    return jsonify({"run": прогон})


@app.get("/api/health")
def api_health():
    инфо = agent.info()
    return jsonify({"ok": True, "model": инфо["model_key"], "layers": инфо["layers"]})


if __name__ == "__main__":
    # Порт и адрес берутся из окружения: 5000 бывает занят — хотя бы прошлым
    # экземпляром этого же сервера. Тогда Flask пишет «Address already in use»
    # и завершается, а браузер продолжает показывать прежний экземпляр со
    # старым кодом; отличить это от поломки трудно.
    порт = int(os.getenv("PORT", "5000"))
    адрес = os.getenv("HOST", "127.0.0.1")
    print(f"Открыть: http://{адрес}:{порт}")
    app.run(debug=True, host=адрес, port=порт)
