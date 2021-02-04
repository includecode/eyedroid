import desktop_notify as dn
class GraphicUI(object):

    def __init__(self):
        pass

    def send_notification(self, title, message):
        notify = dn.Notify(title, message)
        notify.set_timeout(500)
        notify.show()
        return